// TODO: @MaxCompleteAPI, move this into a cpp for the VulkanApplication
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "VulkanApplication.h"

#include <iostream>

// Lorian or Spock

int main()
{
    VulkanApplication app;

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
