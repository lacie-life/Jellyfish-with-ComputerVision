///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2021, STEREOLABS.
//
// All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

/*****************************************************************************************
 ** This sample demonstrates how to detect human bodies and retrieves their 3D position **
 **         with the ZED SDK and display the result in an OpenGL window.                **
 *****************************************************************************************/

// ZED includes
#include <sl/Camera.hpp>

// Sample includes
#include "GLViewer.hpp"
#include "TrackingViewer.hpp"

// Using std and sl namespaces
using namespace std;
using namespace sl;

bool is_playback = false;
void print(string msg_prefix, ERROR_CODE err_code = ERROR_CODE::SUCCESS, string msg_suffix = "");
void parseArgs(int argc, char **argv, InitParameters& param);

int main(int argc, char **argv) {

#ifdef _SL_JETSON_
    const bool isJetson = true;
#else
    const bool isJetson = false;
#endif

    // Create ZED objects
    Camera zed;
    InitParameters init_parameters;
    init_parameters.camera_resolution = RESOLUTION::HD1080;
    // On Jetson the object detection combined with an heavy depth mode could reduce the frame rate too much
    init_parameters.depth_mode = isJetson ? DEPTH_MODE::PERFORMANCE : DEPTH_MODE::ULTRA;
    init_parameters.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;
    parseArgs(argc, argv, init_parameters);

    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("Open Camera", returned_state, "\nExit program.");
        zed.close();
        return EXIT_FAILURE;
    }

    // Enable Positional tracking (mandatory for object detection)
    PositionalTrackingParameters positional_tracking_parameters;
    //If the camera is static, uncomment the following line to have better performances and boxes sticked to the ground.
    //positional_tracking_parameters.set_as_static = true;
    returned_state = zed.enablePositionalTracking(positional_tracking_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("enable Positional Tracking", returned_state, "\nExit program.");
        zed.close();
        return EXIT_FAILURE;
    }

    // Enable the Objects detection module
    ObjectDetectionParameters obj_det_params;
    obj_det_params.enable_tracking = true; // track people across images flow
    obj_det_params.enable_body_fitting = false; // smooth skeletons moves
    obj_det_params.detection_model = isJetson ? DETECTION_MODEL::HUMAN_BODY_FAST : DETECTION_MODEL::HUMAN_BODY_ACCURATE;

    returned_state = zed.enableObjectDetection(obj_det_params);
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("enable Object Detection", returned_state, "\nExit program.");
        zed.close();
        return EXIT_FAILURE;
    }

	auto camera_config = zed.getCameraInformation().camera_configuration;

    // For 2D GUI
    Resolution display_resolution(min((int)camera_config.resolution.width, 1280), min((int)camera_config.resolution.height, 720));
    cv::Mat image_left_ocv(display_resolution.height, display_resolution.width, CV_8UC4, 1);
    Mat image_left(display_resolution, MAT_TYPE::U8_C4, image_left_ocv.data, image_left_ocv.step);
    sl::float2 img_scale(display_resolution.width / (float)camera_config.resolution.width, display_resolution.height / (float) camera_config.resolution.height);
    char key = ' ';

	// 3D View
	Resolution pc_resolution(min((int)camera_config.resolution.width, 720), min((int)camera_config.resolution.height, 404));
	auto camera_parameters = zed.getCameraInformation(pc_resolution).camera_configuration.calibration_parameters.left_cam;
	Mat point_cloud(pc_resolution, MAT_TYPE::F32_C4, MEM::GPU);
	// Create OpenGL Viewer
	GLViewer viewer;
	viewer.init(argc, argv, camera_parameters, obj_det_params.enable_tracking);

	Pose cam_pose;
	cam_pose.pose_data.setIdentity();

    // Configure object detection runtime parameters
    ObjectDetectionRuntimeParameters objectTracker_parameters_rt;
    objectTracker_parameters_rt.detection_confidence_threshold = 40;

    // Create ZED Objects filled in the main loop
    Objects bodies;
	bool quit = false;

    Plane floor_plane; // floor plane handle
    Transform reset_from_floor_plane; // camera transform once floor plane is detected

    // Main Loop
    bool need_floor_plane = positional_tracking_parameters.set_as_static;

	bool gl_viewer_available = true;
    while (gl_viewer_available && !quit && key != 'q') {
        // Grab images
        if (zed.grab() == ERROR_CODE::SUCCESS) {
            // Once the camera has started, get the floor plane to stick the bounding box to the floor plane.
            // Only called if camera is static (see PositionalTrackingParameters)
            if (need_floor_plane) {
                if (zed.findFloorPlane(floor_plane, reset_from_floor_plane) == ERROR_CODE::SUCCESS) {
                    need_floor_plane = false;
                    viewer.setFloorPlaneEquation(floor_plane.getPlaneEquation());
                }
            }

            // Retrieve Detected Human Bodies
            zed.retrieveObjects(bodies, objectTracker_parameters_rt);

            //OCV View
            zed.retrieveImage(image_left, VIEW::LEFT, MEM::CPU, display_resolution);
			zed.retrieveMeasure(point_cloud, MEASURE::XYZRGBA, MEM::GPU, pc_resolution);
			zed.getPosition(cam_pose, REFERENCE_FRAME::WORLD);

			string window_name = "ZED| 2D View";

			//Update GL View
			viewer.updateData(point_cloud, bodies.object_list, cam_pose.pose_data);

			gl_viewer_available = viewer.isAvailable();
			if (is_playback && zed.getSVOPosition() == zed.getSVONumberOfFrames()) {
				quit = true;
			}
			render_2D(image_left_ocv, img_scale, bodies.object_list, obj_det_params.enable_tracking);
			cv::imshow(window_name, image_left_ocv);
			key = cv::waitKey(10);
        }
    }

    // Release objects
	viewer.exit();
	image_left.free();
    point_cloud.free();
    floor_plane.clear();
    bodies.object_list.clear();

    // Disable modules
    zed.disableObjectDetection();
    zed.disablePositionalTracking();
    zed.close();

    return EXIT_SUCCESS;
}

void parseArgs(int argc, char **argv, InitParameters& param) {
    if (argc > 1 && string(argv[1]).find(".svo") != string::npos) {
        // SVO input mode
        param.input.setFromSVOFile(argv[1]);
		is_playback = true;
        cout << "[Sample] Using SVO File input: " << argv[1] << endl;
    } else if (argc > 1 && string(argv[1]).find(".svo") == string::npos) {
        string arg = string(argv[1]);
        unsigned int a, b, c, d, port;
        if (sscanf(arg.c_str(), "%u.%u.%u.%u:%d", &a, &b, &c, &d, &port) == 5) {
            // Stream input mode - IP + port
            string ip_adress = to_string(a) + "." + to_string(b) + "." + to_string(c) + "." + to_string(d);
            param.input.setFromStream(String(ip_adress.c_str()), port);
            cout << "[Sample] Using Stream input, IP : " << ip_adress << ", port : " << port << endl;
        } else if (sscanf(arg.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d) == 4) {
            // Stream input mode - IP only
            param.input.setFromStream(String(argv[1]));
            cout << "[Sample] Using Stream input, IP : " << argv[1] << endl;
        } else if (arg.find("HD2K") != string::npos) {
            param.camera_resolution = RESOLUTION::HD2K;
            cout << "[Sample] Using Camera in resolution HD2K" << endl;
        } else if (arg.find("HD1080") != string::npos) {
            param.camera_resolution = RESOLUTION::HD1080;
            cout << "[Sample] Using Camera in resolution HD1080" << endl;
        } else if (arg.find("HD720") != string::npos) {
            param.camera_resolution = RESOLUTION::HD720;
            cout << "[Sample] Using Camera in resolution HD720" << endl;
        } else if (arg.find("VGA") != string::npos) {
            param.camera_resolution = RESOLUTION::VGA;
            cout << "[Sample] Using Camera in resolution VGA" << endl;
        }
    }
}

void print(string msg_prefix, ERROR_CODE err_code, string msg_suffix) {
    cout << "[Sample]";
    if (err_code != ERROR_CODE::SUCCESS)
        cout << "[Error]";
    cout << " " << msg_prefix << " ";
    if (err_code != ERROR_CODE::SUCCESS) {
        cout << " | " << toString(err_code) << " : ";
        cout << toVerbose(err_code);
    }
    if (!msg_suffix.empty())
        cout << " " << msg_suffix;
    cout << endl;
}
