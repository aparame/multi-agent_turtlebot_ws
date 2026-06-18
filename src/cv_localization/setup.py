from glob import glob
import os

from setuptools import find_packages, setup


package_name = 'cv_localization'


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='adi2440',
    maintainer_email='adiparamesh@gmail.com',
    description='Overhead camera localization and direct MPPI control GUI.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'cv_mppi_direct_gui = cv_localization.cv_mppi_direct_gui:main',
            'cv_rl_direct_controller = cv_localization.cv_rl_direct_controller:main',
            'tb3_vlcm_live_collector = cv_localization.tb3_vlcm_live_collector:main',
        ],
    },
)
