"""Direct CycloneDDS access to a few ROS 2 topics — no rclpy.

We mirror three things by hand so a pure-pip Python env can talk to a
bar_ros2 bringup:

* **Message types** — ``IdlStruct`` dataclasses matching the rosidl-
  generated .idl for ``builtin_interfaces/Time``, ``std_msgs/Header``,
  ``bar_msgs/MITCommand``, and ``sensor_msgs/JointState``.
* **Topic prefix** — ROS ``/foo`` → DDS ``rt/foo``.
* **Type-name namespace** — ROS ``pkg/msg/Name`` is registered as DDS
  ``pkg::msg::dds_::Name_``.

CycloneDDS-python on this side interoperates with either
``rmw_cyclonedds_cpp`` or ``rmw_fastrtps_cpp`` on the bringup — both
are RTPS-over-UDP with CDR encoding. No ``RMW_IMPLEMENTATION``
override required.
"""

from dataclasses import dataclass, field

from cyclonedds.core import Policy, Qos
from cyclonedds.domain import DomainParticipant
from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import float64, int32, sequence, uint32
from cyclonedds.pub import DataWriter, Publisher
from cyclonedds.sub import DataReader, Subscriber
from cyclonedds.topic import Topic
from cyclonedds.util import duration


def ros_topic_to_dds(ros_topic: str) -> str:
    """Map a ROS 2 topic name to its DDS wire name.

    ``/foo/bar`` -> ``rt/foo/bar``. Topics already starting with ``rt/``
    are returned verbatim. rmw_cyclonedds_cpp uses ``rt/`` for data
    topics; ``rq/`` and ``rr/`` for service request/response.
    """
    if ros_topic.startswith("rt/"):
        return ros_topic
    return "rt" + ros_topic if ros_topic.startswith("/") else "rt/" + ros_topic


# ---- IDL mirror types -----------------------------------------------------


@dataclass
class Time(IdlStruct, typename="builtin_interfaces::msg::dds_::Time_"):
    sec: int32 = 0
    nanosec: uint32 = 0


@dataclass
class Header(IdlStruct, typename="std_msgs::msg::dds_::Header_"):
    stamp: Time = field(default_factory=Time)
    frame_id: str = ""


@dataclass
class MITCommand(IdlStruct, typename="bar_msgs::msg::dds_::MITCommand_"):
    header: Header = field(default_factory=Header)
    joint_names: sequence[str] = field(default_factory=list)
    position: sequence[float64] = field(default_factory=list)
    velocity: sequence[float64] = field(default_factory=list)
    effort: sequence[float64] = field(default_factory=list)
    stiffness: sequence[float64] = field(default_factory=list)
    damping: sequence[float64] = field(default_factory=list)


@dataclass
class JointState(IdlStruct, typename="sensor_msgs::msg::dds_::JointState_"):
    header: Header = field(default_factory=Header)
    name: sequence[str] = field(default_factory=list)
    position: sequence[float64] = field(default_factory=list)
    velocity: sequence[float64] = field(default_factory=list)
    effort: sequence[float64] = field(default_factory=list)


# ---- QoS profiles ----------------------------------------------------------


def reliable_keep_last(depth: int) -> Qos:
    """ROS 2 default-ish QoS: reliable, keep-last, volatile.

    Matches the broadcaster's default sensor-rate publisher and the
    RemotePolicyController's subscription (depth 4).
    """
    return Qos(
        Policy.Reliability.Reliable(duration(seconds=1)),
        Policy.History.KeepLast(depth),
        Policy.Durability.Volatile,
    )


# ---- Tiny pub/sub façade ---------------------------------------------------


class DdsContext:
    """Per-process DDS state: one DomainParticipant + shared pub/sub.

    Holds a single ``DomainParticipant`` for the given domain id (matches
    ``ROS_DOMAIN_ID``) and a single ``Publisher`` / ``Subscriber`` reused
    across all topics. Mirrors Lite-SDK2's factory pattern.
    """

    def __init__(self, domain_id: int = 0) -> None:
        self.participant = DomainParticipant(domain_id)
        self.publisher = Publisher(self.participant)
        self.subscriber = Subscriber(self.participant)

    def writer(self, ros_topic: str, msg_type: type, qos: Qos) -> DataWriter:
        topic = Topic(self.participant, ros_topic_to_dds(ros_topic), msg_type, qos=qos)
        return DataWriter(self.publisher, topic, qos=qos)

    def reader(self, ros_topic: str, msg_type: type, qos: Qos) -> DataReader:
        topic = Topic(self.participant, ros_topic_to_dds(ros_topic), msg_type, qos=qos)
        return DataReader(self.subscriber, topic, qos=qos)
