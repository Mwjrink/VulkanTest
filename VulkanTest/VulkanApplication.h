#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <sys/stat.h>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <stdexcept>
#include <vector>

#include <iostream>

#include "logger.h"

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                      const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger)
{
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr)
    {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    }
    else
    {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger,
                                   const VkAllocationCallbacks* pAllocator)
{
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr)
    {
        func(instance, debugMessenger, pAllocator);
    }
}

struct QueueFamilyIndices
{
    std::optional<uint32_t> graphicsFamily;

    bool isComplete() { return graphicsFamily.has_value(); }
};

class VulkanApplication
{
  private:
    VkInstance instance;

    VkDebugUtilsMessengerEXT debugMessenger;

    GLFWwindow* window;
    const int   WIDTH  = 800;
    const int   HEIGHT = 600;

    std::ofstream graphicsLogFile;
    Logger        graphicsLog;

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;

	VkDevice device;

    const std::vector<const char*> validationLayers = {
        "VK_LAYER_LUNARG_standard_validation"};  // {"VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT      messageSeverity,
                                                        VkDebugUtilsMessageTypeFlagsEXT             messageType,
                                                        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                                        void*                                       pUserData)
    {
        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

        // if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        //      {
        //          // Message is important enough to show
        // VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT // Diagnostic message
        // VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT	   // Informational message like the creation of a resource
        // VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT // Message about behavior that is not necessarily an error,
        //												   // but very likely a bug in your application
        // VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT   // Message about behavior that is invalid and may cause crashes
        //      }

        // VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT     // Some event has happened that is unrelated to the specification
        //												   // or performance
        // VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT  // Something has happened that violates the specification or
        //												   // indicates a possible mistake
        // VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT // Potential non-optimal use of Vulkan

        return VK_FALSE;
    }

    void initLog(std::string logPath)
    {
        std::string fileName = "graphics0.log";

        int logNumber = 0;
        while (std::filesystem::exists(fileName))
        {
            fileName = "grapics" + std::to_string(logNumber) + ".log";
            logNumber++;
        }

        graphicsLogFile.open(logPath + "/Graphics/" + fileName);

        graphicsLog = Logger(&graphicsLogFile);
    }

    void initWindow()
    {
        glfwInit();
        // Do not create an OpenGL Context
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        // TODO: for now because this is complicated
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        // create the window and store a pointer to it
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    }

    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo)
    {
        createInfo                 = {};
        createInfo.sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                     VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                     VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
        createInfo.pUserData       = nullptr;  // Optional
    }

    void setupDebugMessenger()
    {
        if (!enableValidationLayers) return;

        VkDebugUtilsMessengerCreateInfoEXT createInfo;
        populateDebugMessengerCreateInfo(createInfo);

        auto result = CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger);
        throwOnError(result, "failed to set up debug messenger!");
    }

    void initVulkan()
    {
        createInstance();
        setupDebugMessenger();
        pickPhysicalDevice();
        createLogicalDevice();
    }

    void createLogicalDevice() {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        VkDeviceQueueCreateInfo queueCreateInfo = {};
        queueCreateInfo.sType                   = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex        = indices.graphicsFamily.value();
        queueCreateInfo.queueCount              = 1;

		float queuePriority              = 1.0f;
        queueCreateInfo.pQueuePriorities = &queuePriority;
	}

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device)
    {
        QueueFamilyIndices indices;

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        int i = 0;
        for (const auto& queueFamily : queueFamilies)
        {
            if (queueFamily.queueCount > 0 && queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
            {
                indices.graphicsFamily = i;
            }

			// bleh

            if (indices.isComplete())
            {
                break;
            }

            i++;
        }

        return indices;
    }

    void pickPhysicalDevice()
    {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        if (deviceCount == 0)
        {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        // Use an ordered map to automatically sort candidates by increasing score
        std::multimap<int, VkPhysicalDevice> candidates;

        for (const auto& device : devices)
        {
            int score = rateDeviceSuitability(device);
            candidates.insert(std::make_pair(score, device));
        }

        // Check if the best candidate is suitable at all
        if (candidates.rbegin()->first > 0)
        {
            physicalDevice = candidates.rbegin()->second;
        }
        else
        {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }

    int rateDeviceSuitability(VkPhysicalDevice device)
    {
        VkPhysicalDeviceProperties deviceProperties;
        VkPhysicalDeviceFeatures   deviceFeatures;
        vkGetPhysicalDeviceProperties(device, &deviceProperties);
        vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

        int score = 0;

        // Discrete GPUs have a significant performance advantage
        if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        {
            score += 1000;
        }

        QueueFamilyIndices indices = findQueueFamilies(device);
        if (!indices.isComplete()) return 0;

        // Maximum possible size of textures affects graphics quality
        // score += deviceProperties.limits.maxImageDimension2D;

        // Application can't function without geometry shaders
        // if (!deviceFeatures.geometryShader)
        //{
        //    return 0;
        //}

        return score;
    }

    void createInstance()
    {
        if (enableValidationLayers && !checkValidationLayerSupport())
        {
            throw std::runtime_error("validation layers requested, but not available!");
        }

        VkApplicationInfo appInfo  = {};
        appInfo.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName   = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName        = "No Engine";
        appInfo.engineVersion      = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion         = VK_API_VERSION_1_0;

        VkInstanceCreateInfo createInfo = {};
        createInfo.sType                = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo     = &appInfo;

        auto extensions                    = getRequiredExtensions();
        createInfo.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo;
        if (enableValidationLayers)
        {
            createInfo.enabledLayerCount   = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();

            populateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
        }
        else
        {
            createInfo.enabledLayerCount = 0;

            createInfo.pNext = nullptr;
        }

        VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);

        throwOnError(result, "failed to create instance!");
    }

    bool checkValidationLayerSupport()
    {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        graphicsLog.LogHeader("Requesting Vulkan validation layers\t [" + std::to_string(layerCount) + "]");

        graphicsLog.Log("Layer", "Found Status", -12);
        graphicsLog.LogMaxWidth('-');

        bool missingLayers = false;
        for (auto layerName : validationLayers)
        {
            bool              layerFound = false;
            VkLayerProperties layer_info;
            for (const auto& layerProperties : availableLayers)
            {
                if (strcmp(layerName, layerProperties.layerName) == 0)
                {
                    layerFound = true;
                    layer_info = layerProperties;
                    break;
                }
            }

            if (!layerFound)
            {
                missingLayers = true;
                // return false;
                PrintLayerStatus(layerName, layerFound);
            }
            else
            {
                PrintLayerStatus(layer_info, layerName, layerFound);
            }
        }

        graphicsLog.Log();

        return !missingLayers;
        // return true;
    }

    void PrintLayerStatus(VkLayerProperties layer_info, std::string layer_name, bool layer_found)
    {
        std::string major   = std::to_string(VK_VERSION_MAJOR(layer_info.specVersion));
        std::string minor   = std::to_string(VK_VERSION_MINOR(layer_info.specVersion));
        std::string patch   = std::to_string(VK_VERSION_PATCH(layer_info.specVersion));
        std::string version = major + "." + minor + "." + patch;

        std::string mark = (layer_found) ? std::string(CHECK) : std::string(CROSS);
        graphicsLog.Log();
        graphicsLog.Log(std::string(layer_name) + ", Vulkan version " + version + ", layer version " +
                            std::to_string(layer_info.implementationVersion),
                        "[" + mark + "]", -3, '.');

        if (layer_found)
        {
            graphicsLog.Log("\tDescription:", std::string(layer_info.description),
                            -int(std::string(layer_info.description).length()));
        }
    }

    void PrintLayerStatus(std::string layer_name, bool layer_found)
    {
        std::string mark = (layer_found) ? std::string(CHECK) : std::string(CROSS);
        graphicsLog.Log();
        graphicsLog.Log(std::string(layer_name), "[" + mark + "]", -3, '.');
    }

    std::vector<const char*> getRequiredExtensions()
    {
        uint32_t     glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        if (enableValidationLayers)
        {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return extensions;
    }

    void check_extensions()
    {
        uint32_t extensionCount = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> extensions(extensionCount);

        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

        graphicsLog.Log("available extensions:");

        for (const auto& extension : extensions)
        {
            graphicsLog.Log(extension.extensionName, 4);
        }

        graphicsLog.Log();
    }

    void throwOnError(VkResult result, std::string message)
    {
        if (result != VK_SUCCESS)
        {
            throw std::runtime_error(message);
        }
    }

    void mainLoop()
    {
        while (!glfwWindowShouldClose(window))
        {
            glfwPollEvents();
        }
    }

    void cleanup()
    {
        if (enableValidationLayers)
        {
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }

        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);

        glfwTerminate();

        graphicsLogFile.flush();
        graphicsLogFile.close();
    }

  public:
    void run()
    {
        try
        {
            initLog("A:/Development/Vulkan Project/VulkanTest/x64/Debug");
            initWindow();
            initVulkan();
            check_extensions();
            mainLoop();
        }
        catch (const std::exception& e)
        {
            graphicsLog.Flush();
            cleanup();
            throw e;
        }

        cleanup();
    }
};
