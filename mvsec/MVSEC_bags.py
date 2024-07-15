import bag_indexer
import numpy as np
from rosbag import Bag

def read_bag_odom(bag_path):
    left_sync_topics = (['/davis/left/odometry','/davis/left/depth_image_raw','/davis/left/depth_image_rect'],[0.05,0.05,0.05])
    bag = bag_indexer.get_bag_indexer(bag_path, [left_sync_topics])
    left_cam_readers = {}

    for t in left_sync_topics[0]:
        if 'image' in t:
            left_cam_readers[t] = None #self.bag.get_image_topic_reader(t)
        else:
            left_cam_readers[t] = bag.get_topic_reader(t)
            
    return left_cam_readers['/davis/left/odometry']

def p_q_t_from_msg(msg):
        p = np.array([msg.pose.position.x, 
                      msg.pose.position.y, 
                      msg.pose.position.z])
        q = np.array([msg.pose.orientation.x,
                      msg.pose.orientation.y, 
                      msg.pose.orientation.z, 
                      msg.pose.orientation.w])
        t = msg.header.stamp.to_sec()
        return p, q, t
    
if __name__ == '__main__':
    bag_path = 'D:\\AI\\visual\\mv\\outdoor_day1_gt.bag'
    
    bag =Bag(bag_path)
    start_time = bag.get_start_time()
    
    msg = read_bag_odom(bag_path)
    odom = msg[0].message
    p,q,t = p_q_t_from_msg(odom)