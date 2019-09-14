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
