from setuptools import find_packages, setup

package_name = 'move_panda'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dheeraj',
    maintainer_email='bogishettydheeraj007@gmail.com',
    description='ROS2 package for MoveXYZW action client',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'move_panda_client = move_panda.move_panda_action_client:main'
        ],
    },
)
