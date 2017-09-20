# encode: utf-8
# python2

import rospy
from std_msgs.msg import String
import subprocess

run = ['./darknet', 'detector', 'demo', 'cfg/coco.data', 'cfg/yolo.cfg', 'yolo.weights', '/data/170912-02/camera_0.avi']
run = run[:-1]
child = subprocess.Popen(run, stdout=subprocess.PIPE)
pub = rospy.Publisher('coodinate', String, queue_size=100)
rospy.init_node('talker', anonymous=True)
print("start...")

while True:
    l = child.stdout.readline()
    if l.startswith('>>'):
        p_str = String()
        p_str.data = l
        pub.publish(p_str)
    elif not l:
        break

print("over")
