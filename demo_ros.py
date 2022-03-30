#!/usr/bin/env python
# -*- coding:utf-8 -*-

# Copyright (c) 2022 SoftBank Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import math

import cv2
import message_filters
import numpy as np
import rospkg
import rospy
import sensor_msgs.point_cloud2 as pc2
import torch
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import ColorRGBA
from tfpose_ros.msg import BodyPartElm, Person, Persons
from spencer_tracking_msgs.msg import DetectedPersons, DetectedPerson
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses


class Estimator(object):
    def __init__(self):

        pkg = rospkg.RosPack().get_path('lightweight_pose_estimation')
        checkpoint_path = rospy.get_param('~checkpoint', pkg + '/models/checkpoint_iter_370000.pth')
        self.height = rospy.get_param('~height', 480)
        self.cpu = rospy.get_param('~cpu', False)
        self.track = rospy.get_param('~track', True)
        self.smooth = rospy.get_param('~smooth', True)
        self.__detection_id_increment = rospy.get_param('~detection_id_increment', 1)
        self.__last_detection_id = rospy.get_param('~detection_id_offset', 0)
        self._bridge = CvBridge()

        self.previous_poses = []

        net = PoseEstimationWithMobileNet()
        self.net = net.eval()
        if not self.cpu:
            self.net = self.net.cuda()
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        load_state(self.net, checkpoint)

        color_sub = message_filters.Subscriber("~color", Image)
        points_sub = message_filters.Subscriber("~points", PointCloud2)
        sub = message_filters.ApproximateTimeSynchronizer([color_sub, points_sub], 10, 0.1)
        sub.registerCallback(self.inference)

        self._pub = rospy.Publisher('~output', Image, queue_size=1)
        self.__pub_keypoints = rospy.Publisher('~persons', Persons, queue_size=1)
        self.__pub_markers = rospy.Publisher('~markers', MarkerArray, queue_size=1)
        self.__pub_pose = rospy.Publisher('~poses', DetectedPersons, queue_size=1)

    def inference(self, color, points):
        img = self._bridge.imgmsg_to_cv2(color, 'bgr8')

        stride = 8
        upsample_ratio = 4
        num_keypoints = Pose.num_kpts

        orig_img = img.copy()
        heatmaps, pafs, scale, pad = self.infer_fast(img, stride, upsample_ratio)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)

        if self.track:
            track_poses(self.previous_poses, current_poses, smooth=self.smooth)
            self.previous_poses = current_poses
        for pose in current_poses:
            pose.draw(img)
        img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
        for pose in current_poses:
            cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                          (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
            if self.track:
                cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))

        result = self._bridge.cv2_to_imgmsg(img, 'bgr8')
        result.header = color.header
        self._pub.publish(result)

        msg = self.__poses_to_msg(current_poses, points)
        msg.header = points.header
        self.__pub_keypoints.publish(msg)
        self.__pub_markers.publish(self.__to_markers(msg))
        self.__pub_pose.publish(self.__to_spencer_msg(msg))

    def infer_fast(self, img, stride, upsample_ratio,
                pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
        height, _, _ = img.shape
        scale = self.height / height

        scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        scaled_img = normalize(scaled_img, img_mean, img_scale)
        min_dims = [self.height, max(scaled_img.shape[1], self.height)]
        padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
        if not self.cpu:
            tensor_img = tensor_img.cuda()

        stages_output = self.net(tensor_img)

        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

        return heatmaps, pafs, scale, pad


    def __poses_to_msg(self, poses, points):
        persons = Persons()
        height, width = points.height, points.width

        for pose in poses:
            person = Person()

            for i, name in enumerate(pose.kpt_names):
                keypoint = pose.keypoints[i]

                body_part_msg = BodyPartElm()
                body_part_msg.part_id = i
                x, y = np.round(keypoint[0]), np.round(keypoint[1])
                expand = 3
                centers = [[xx, yy] for yy in range(max(int(y - expand), 0), min(int(y + expand), height))
                           for xx in range(max(int(x - expand), 0), min(int(x + expand), width))]
                if i in range(14, 18):
                    centers = [[int(x), int(y)]]
                pts = [p for p in pc2.read_points(points, ('x', 'y', 'z'), uvs=centers, skip_nans=True)]
                if not pts:
                    continue
                pt = np.mean(pts, axis=0)
                body_part_msg.x = pt[0]
                body_part_msg.y = pt[1]
                body_part_msg.z = pt[2]
                body_part_msg.confidence = pose.confidence
                person.body_part.append(body_part_msg)
            persons.persons.append(person)

        return persons

    def __to_markers(self, keypoints):
        markers = MarkerArray()

        links = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11),
                 (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
        CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
                    [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
                    [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
        markers.markers.append(Marker(header=keypoints.header, action=Marker.DELETEALL))
        for i, p in enumerate(keypoints.persons):
            body_parts = [None] * 18
            for k in p.body_part:
                body_parts[k.part_id] = k

            marker = Marker()
            marker.header = keypoints.header
            marker.ns = 'person_{}'.format(i)
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD
            marker.scale.x = 0.05
            marker.lifetime = rospy.Duration(1)
            id = 0
            for ci, link in enumerate(links):
                if body_parts[link[0]] is not None and body_parts[link[1]] is not None:
                    marker.points.append(Point(body_parts[link[0]].x,
                                               body_parts[link[0]].y,
                                               body_parts[link[0]].z))
                    marker.points.append(Point(body_parts[link[1]].x,
                                               body_parts[link[1]].y,
                                               body_parts[link[1]].z))
                    color = CocoColors[ci]
                    marker.id = id
                    id += 1
                    marker.colors.append(ColorRGBA(float(color[0]) / 255,
                                                   float(color[1]) / 255,
                                                   float(color[2]) / 255,
                                                   1.0))
                    marker.colors.append(ColorRGBA(float(color[0]) / 255,
                                                   float(color[1]) / 255,
                                                   float(color[2]) / 255,
                                                   1.0))
            markers.markers.append(marker)

        return markers

    def __to_spencer_msg(self, keypoints):
        persons = DetectedPersons()
        persons.header = keypoints.header
        for p in keypoints.persons:
            for k in p.body_part:
                if k.part_id in [1, 8, 11]:
                    person = DetectedPerson()
                    person.modality = DetectedPerson.MODALITY_GENERIC_RGBD
                    person.pose.pose.position.x = k.x
                    person.pose.pose.position.y = k.y
                    person.pose.pose.position.z = k.z
                    person.confidence = k.confidence
                    person.detection_id = self.__last_detection_id
                    self.__last_detection_id += self.__detection_id_increment
                    large_var = 999999999
                    pose_variance = 0.05
                    person.pose.covariance[0 * 6 + 0] = pose_variance
                    person.pose.covariance[1 * 6 + 1] = pose_variance
                    person.pose.covariance[2 * 6 + 2] = pose_variance
                    person.pose.covariance[3 * 6 + 3] = large_var
                    person.pose.covariance[4 * 6 + 4] = large_var
                    person.pose.covariance[5 * 6 + 5] = large_var
                    persons.detections.append(person)
                    break

        return persons

def normalize(img, img_mean, img_scale):
    img = np.array(img, dtype=np.float32)
    img = (img - img_mean) * img_scale
    return img


def pad_width(img, stride, pad_value, min_dims):
    h, w, _ = img.shape
    h = min(min_dims[0], h)
    min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
    min_dims[1] = max(min_dims[1], w)
    min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
    pad = []
    pad.append(int(math.floor((min_dims[0] - h) / 2.0)))
    pad.append(int(math.floor((min_dims[1] - w) / 2.0)))
    pad.append(int(min_dims[0] - h - pad[0]))
    pad.append(int(min_dims[1] - w - pad[1]))
    padded_img = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3],
                                    cv2.BORDER_CONSTANT, value=pad_value)
    return padded_img, pad

if __name__ == '__main__':
    rospy.init_node('pose_estimator')
    _ = Estimator()
    rospy.spin()
