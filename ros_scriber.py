# encode: utf-8
# python2

import rospy
from std_msgs.msg import String

def callback(msg):
    rospy.loginfo(msg.data)
    # print(msg.data)

# rospy.init_node('coodinate')
rospy.init_node('listener')
sub = rospy.Subscriber('coodinate', String, callback)
print('listening...')
rospy.spin()
