ARG DEBIAN_FRONTEND=noninteractive
FROM gitlab.lrz.de:5005/av2.0/av_software/autoware/microservice/base:cuda-humble-x86_64-lt
ARG ROS_DISTRO=humble

#multiproject
ARG UNIVERSE_COMMIT_HASH

# Set environment variables
ENV ROS_DISTRO=${ROS_DISTRO}

# Create a non-root user
ARG USERNAME=tum
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
  && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
  # [Optional] Add sudo support for the non-root user
  && apt-get update \
  && apt-get install -y sudo \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME\
  && chmod 0440 /etc/sudoers.d/$USERNAME \
  # Cleanup
  && rm -rf /var/lib/apt/lists/* \
  && echo "source /usr/share/bash-completion/completions/git" >> /home/$USERNAME/.bashrc \
  && echo "if [ -f /opt/ros/${ROS_DISTRO}/setup.bash ]; then source /opt/ros/${ROS_DISTRO}/setup.bash; fi" >> /home/$USERNAME/.bashrc
# Set up auto-source of workspace for tum user
ARG WORKSPACE
RUN echo "if [ -f ${WORKSPACE}/install/setup.bash ]; then source ${WORKSPACE}/install/setup.bash; fi" >> /home/tum/.bashrc

WORKDIR /dev_ws/src
COPY centerpoint.repos /dev_ws/src/centerpoint.repos

# Import source code
RUN vcs import . < /dev_ws/src/centerpoint.repos

# Build workspace
# RUN bash -c "source /opt/ros/$ROS_DISTRO/setup.bash && \
#              colcon build --packages-up-to \
#              lidar_centerpoint --cmake-args -DCMAKE_BUILD_TYPE=Release"

# RUN echo '#!/usr/bin/env bash' > /ros_entrypoint.sh
# RUN echo 'source /opt/ros/${ROS_DISTRO}/setup.bash' >> /ros_entrypoint.sh
# RUN echo '. /autoware/install/setup.bash' >> /ros_entrypoint.sh
# RUN echo 'export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp' >> /ros_entrypoint.sh
# RUN echo 'exec "$@"' >> /ros_entrypoint.sh
# RUN chmod +x /ros_entrypoint.sh

RUN bash -c "rm -rf /dev_ws/src/universe/autoware.universe/perception/lidar_centerpoint"

# ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
