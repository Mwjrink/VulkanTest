#ifdef NDEBUG
#define DEBUG_OPTIONS
#endif

#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "Interfaces.h"
#include "Logger.h"
#include "VulkanApplication.h"

void update(float dt);

extern RenderGroup renderGroup;
extern Instance    instance;
extern Model       model;

std::ofstream graphicsLogFile;
void          initGraphicsLog()
{
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
    auto graphicsLog = Logger(&graphicsLogFile);

    // TODO: @Max, make this a singleton?
    // TODO: @Max, remove all window stuff from here and have this just be vulkan/rendering
    auto app = VulkanApplication(graphicsLog, 1920, 1080, "Vulkan Fun!", false);

    renderGroup = RenderGroup(&app);
    model       = renderGroup.addNewModel("models/chalet.obj", "textures/chalet.jpg");
    instance    = model.addInstance();

    // create a camera class to handle this
    auto view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));

    app.setViewMatrix(view);
    app.setFov(45.0f);

    try
    {
        {
            // auto  oldTime = glfwGetTime();
            // float dt      = oldTime;
            // while (!glfwWindowShouldClose(app.window))
            // {
            //     auto current = glfwGetTime();
            //     dt           = current - oldTime;
            //     oldTime      = current;

            //     // processInput();

            //     update(dt);

            //     app.renderFrame();

            //     glfwPollEvents();
            // }

            // app.waitIdle();
            // app.cleanupApp();
        }

        app.run(update);
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

void update(float dt)
{
    // Not sure if it matters where this goes
    glfwPollEvents();
    // processInput();

    auto modelMarix = instance.ModelMatrix();
    *modelMarix     = glm::rotate(*modelMarix, dt * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
}