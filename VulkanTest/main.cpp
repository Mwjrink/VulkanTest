#ifdef NDEBUG
#define DEBUG_OPTIONS
#endif

#include <iostream>
#include <sstream>

#include "Logger.h"
#include "VulkanApplication.h"

// Lorian or Spock

std::ofstream graphicsLogFile;
void initGraphicsLog(){
    std::string fileName;
    
    {
        // auto t  = std::time(nullptr);
        // auto tm = *std::localtime(&t);
        
        // std::stringstream filenameStream;
        // filenameStream << "graphics" << std::put_time(&tm, "%d-%m-%Y");
        
        fileName = "graphics";
        // filenameStream.str();
    }
    
    int         logNumber = 0;
    std::string file      = "graphics" + fileName + std::to_string(logNumber) + ".log";
    
#ifdef _WIN32
    // TODO: @MaxWindowsSpecific, find a way to make the directory if it does not alreay exist (probably add an ini
    // option for the path) also add a date-timestamp to this
    while (std::filesystem::exists("Graphics/" + file))
    {
        logNumber++;
        file = "graphics" + fileName + std::to_string(logNumber) + ".log";
    }
#elif __APPLE__
    // TODO: @MaxAppleSupport, get a proper path to log to
    // also add a date-timestamp to this
    file = "/Users/maxrink/Development/Vulkan Project/VulkanTest/Graphics/" + fileName + std::to_string(logNumber) + ".log";
#endif
    
    graphicsLogFile.open("Graphics/" + file);
    // TODO: @MaxTemporary, this should be a new file
    graphicsLogFile.clear();
}

int main()
{
    initGraphicsLog();
    Logger graphicsLog(&graphicsLogFile);
    VulkanApplication app(graphicsLog, 1920, 1080, "Vulkan Fun!", false);
    
    std::vector<int> renderGroupIndices;
    renderGroupIndices.push_back(app.createRenderGroup());
    auto modelIndex = app.addModel(renderGroupIndices[0], "models/chalet.obj", "textures/chalet.jpg");
    auto instanceIndex = app.addInstance(renderGroupIndices[0], modelIndex, glm::mat4(1.0f));
    
    auto model = glm::mat4(1.0f);
    
    auto view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    
    // TODO: @MaxCompleteAPI, probably have this auto set from VulkanApplication.h and choose/set FOV
    auto proj = glm::perspective(glm::radians(45.0f), 1920.0f / 1080.0f, 0.1f, 10.0f);
    
    // GLM was originally designed for OpenGL, where the Y coordinate of the clip coordinates is inverted. The easiest
    // way to compensate for that is to flip the sign on the scaling factor of the Y axis in the projection matrix. If
    // you don't do this, then the image will be rendered upside down.
    proj[1][1] *= -1;

    app.setMatrices(view, proj);
    
    
    auto oldTime = glfwGetTime();
    float dt = oldTime;
    try
    {
        while (!glfwWindowShouldClose(app.window))
        {
            auto current = glfwGetTime();
            dt = current - oldTime;
            oldTime = current;
            
            //processInput();
            //Update(dt); {
            
            model = glm::rotate(model, dt * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
            // this is not ideal, probably pass back a pointer/reference to the model matrix in the vector itself to allow the user to modify it
            app.updateInstanceModelMatrix(renderGroupIndices[0], modelIndex, instanceIndex, model);
            
            // } // Update
            
            app.renderFrame(renderGroupIndices);
            
            glfwPollEvents();
        }
        
        app.waitIdle();
        app.cleanupApp();
        //app.run();
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
