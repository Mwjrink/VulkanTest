#ifdef NDEBUG
#define DEBUG_OPTIONS
#endif

#include <iostream>
#include <sstream>

#include "Logger.h"
#include "VulkanApplication.h"

// Lorian or Spock

int main()
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
    logPath = "/Users/maxrink/Development/Vulkan Project/VulkanTest";
#endif

    std::ofstream graphicsLogFile;
    graphicsLogFile.open("Graphics/" + file);
    // TODO: @MaxTemporary, this should be a new file
    graphicsLogFile.clear();

    Logger graphicsLog(&graphicsLogFile);

    VulkanApplication app(graphicsLog, 1920, 1080, "Vulkan Fun!", false, false);

    try
    {
        app.run();
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
