#pragma once

#define GLFW_INCLUDE_VULKAN
//#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <stb_image.h>

#include <algorithm>
#include <array>
#include <cstdint>  // Necessary for UINT32_MAX
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

#include <iostream>

#ifdef _WIN32
#include <filesystem>
#include <optional>
#elif __APPLE__
#include <experimental/optional>
#endif

#include "logger.h"

const int MAX_FRAMES_IN_FLIGHT = 2;

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

struct Vertex
{
    glm::vec2 pos;
    glm::vec3 color;

    static VkVertexInputBindingDescription getBindingDescription()
    {
        VkVertexInputBindingDescription bindingDescription = {};
        bindingDescription.binding                         = 0;
        bindingDescription.stride                          = sizeof(Vertex);
        bindingDescription.inputRate                       = VK_VERTEX_INPUT_RATE_VERTEX;

        // inputRate parameter can have one of the following values:
        // VK_VERTEX_INPUT_RATE_VERTEX:   // Move to the next data entry after each vertex
        // VK_VERTEX_INPUT_RATE_INSTANCE: // Move to the next data entry after each instance

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions()
    {
        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions = {};

        attributeDescriptions[0].binding  = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format   = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[0].offset   = offsetof(Vertex, pos);

        // The format parameter describes the type of data for the attribute. A bit confusingly, the formats are specified
        // using the same enumeration as color formats. The following shader types and formats are commonly used together:
        // float: VK_FORMAT_R32_SFLOAT
        // vec2: VK_FORMAT_R32G32_SFLOAT
        // vec3: VK_FORMAT_R32G32B32_SFLOAT
        // vec4: VK_FORMAT_R32G32B32A32_SFLOAT

        // ivec2: VK_FORMAT_R32G32_SINT, a 2-component vector of 32-bit signed integers
        // uvec4: VK_FORMAT_R32G32B32A32_UINT, a 4-component vector of 32-bit unsigned integers
        // double: VK_FORMAT_R64_SFLOAT, a double-precision (64-bit) float

        attributeDescriptions[1].binding  = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format   = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset   = offsetof(Vertex, color);

        return attributeDescriptions;
    }
};

struct QueueFamilyIndices
{
#ifdef _WIN32
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() { return graphicsFamily.has_value() && presentFamily.has_value(); }
#elif __APPLE__
    std::experimental::optional<uint32_t> graphicsFamily;
    std::experimental::optional<uint32_t> presentFamily;

    bool       isComplete() { return graphicsFamily.operator bool() && presentFamily.operator bool(); }
#endif
};

struct SwapChainSupportDetails
{
    VkSurfaceCapabilitiesKHR        capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR>   presentModes;
};

// Vulkan expects this to be aligned a certain way (see footnote)
// Scalars have to be aligned by N (= 4 bytes given 32 bit floats).
// A vec2 must be aligned by 2N (= 8 bytes)
// A vec3 or vec4 must be aligned by 4N (= 16 bytes)
// A nested structure must be aligned by the base alignment of its members rounded up to a multiple of 16.
// A mat4 matrix must have the same alignment as a vec4.
struct UniformBufferObject
{
    // being explicit about the alignment, though this is not necessarily necessary
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

static std::vector<char> readFile(const std::string& filename)
{
    // We start by opening the file with two flags:
    // ate:	// Start reading at the end of the file
    // binary: // Read the file as binary file(avoid text transformations)

    // The advantage of starting to read at the end of the file is that we can use the read position to determine the size of
    // the file and allocate a buffer:
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open())
    {
        // TODO: @MaxCompleteAPI, log the errors produced here
        throw std::runtime_error("failed to open file!");
    }

    size_t            fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);

    // seek back to the beginning of the file
    file.seekg(0);
    // read all of the bytes at once
    file.read(buffer.data(), fileSize);

    file.close();

    return buffer;
}

class VulkanApplication
{
  private:
    VkInstance instance;

    VkDebugUtilsMessengerEXT debugMessenger;

    VkSurfaceKHR surface;

    GLFWwindow* window;

    // TODO: @MaxCompleteAPI, these should not be constants
    // TODO: @MaxCompleteAPI, make a simple configurations class that stores configs in a .ini or some such file (maybe allow
    //			it to be encrypted) and you can update those values and read them so they persist accross runs (for settings
    //			and stuff)
    int WIDTH  = 800;
    int HEIGHT = 600;

    std::ofstream graphicsLogFile;
    Logger        graphicsLog;

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;

    VkDevice device;
    VkQueue  graphicsQueue;
    VkQueue  presentQueue;

    VkSwapchainKHR swapChain;

    // TODO: @MaxCompleteApi, probably make this an std::array
    std::vector<VkImage> swapChainImages;

    VkFormat   swapChainImageFormat;
    VkExtent2D swapChainExtent;

    std::vector<VkImageView> swapChainImageViews;

    VkRenderPass          renderPass;
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout      pipelineLayout;

    VkPipeline graphicsPipeline;

    std::vector<VkFramebuffer> swapChainFramebuffers;

    VkCommandPool commandPool;

    std::vector<VkCommandBuffer> commandBuffers;

    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence>     inFlightFences;
    size_t                   currentFrame = 0;

    VkBuffer       vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VkBuffer       indexBuffer;
    VkDeviceMemory indexBufferMemory;

    // We need multiple because we need one per in-flight frame (model/projection etc could (vast majority of cases) be
    // different every frame
    std::vector<VkBuffer>       uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;

    VkDescriptorPool             descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;

    VkImage        textureImage;
    VkDeviceMemory textureImageMemory;

    const std::vector<Vertex> vertices = {{{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},  //
                                          {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},   //
                                          {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},    //
                                          {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}};

    // It is possible to use either uint16_t or uint32_t for your index buffer depending on the number of entries in
    // vertices. We can stick to uint16_t for now because we're using less than 65535 unique vertices.
    const std::vector<uint16_t> indices = {0, 1, 2, 2, 3, 0};

    bool framebufferResized = false;

    // cant seem to find the macro for this
    const std::vector<const char*> validationLayers = {"VK_LAYER_KHRONOS_validation", "VK_LAYER_LUNARG_monitor",
                                                       "VK_LAYER_LUNARG_assistant_layer"};

    const std::vector<const char*> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif

    // TODO: @MaxCompleteAPI, pass in a Logger instead of creating it here
    void initLog(std::string logPath)
    {
        std::string fileName = "graphics0.log";

#ifdef _WIN32
        // TODO: @MaxWindowsSpecific, find a way to make the directory if it does not alreay exist (probably add an ini
        // option for the path) also add a date-timestamp to this
        int logNumber = 0;
        while (std::filesystem::exists(logPath + "/Graphics/" + fileName))
        {
            fileName = "graphics" + std::to_string(logNumber) + ".log";
            logNumber++;
        }
#elif __APPLE__
        // TODO: @MaxAppleSupport, get a proper path to log to
        // also add a date-timestamp to this
        logPath = "/Users/maxrink/Development/Vulkan Project/VulkanTest";
#endif

        graphicsLogFile.open(logPath + "/Graphics/" + fileName);
        graphicsLogFile.clear();

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
        // TODO @MaxYikes, I mean this is cool but wtf, code: 17894375109
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    }

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height)
    {
        // TODO: @MaxYikes, see code: 17894375109
        auto app                = reinterpret_cast<VulkanApplication*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT      messageSeverity,
                                                        VkDebugUtilsMessageTypeFlagsEXT             messageType,
                                                        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                                        void*                                       pUserData)
    {
        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

        // TODO: @MaxProperErrorChecking, uncomment and use this nonsense
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
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandPool();
        createTextureImage();
        createVertexBuffer();
        createIndexBuffer();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
        createSyncObjects();
    }

    void createTextureImage()
    {
        int          texWidth, texHeight, texChannels;
        stbi_uc*     pixels    = stbi_load("textures/texture.jpg", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        VkDeviceSize imageSize = texWidth * texHeight * 4;

        if (!pixels)
        {
            graphicsLog.Log("failed to load texture image!");
            throw std::runtime_error("failed to load texture image!");
        }

        VkBuffer       stagingBuffer;
        VkDeviceMemory stagingBufferMemory;

        createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer,
                     stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
        memcpy(data, pixels, static_cast<size_t>(imageSize));
        vkUnmapMemory(device, stagingBufferMemory);

        stbi_image_free(pixels);

        createImage(texWidth, texHeight, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    textureImage, textureImageMemory);
    }

    void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage,
                     VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory)
    {
        VkImageCreateInfo imageInfo = {};
        imageInfo.sType             = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType         = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width      = width;
        imageInfo.extent.height     = height;
        imageInfo.extent.depth      = 1;
        imageInfo.mipLevels         = 1;
        imageInfo.arrayLayers       = 1;
        imageInfo.format            = format;
        imageInfo.tiling            = tiling;
        imageInfo.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage             = usage;
        imageInfo.samples           = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode       = VK_SHARING_MODE_EXCLUSIVE;

        // The tiling field can have one of two values:
        //
        // VK_IMAGE_TILING_LINEAR: Texels are laid out in row-major order like our pixels array
        // VK_IMAGE_TILING_OPTIMAL: Texels are laid out in an implementation defined order for optimal access
        //
        // Unlike the layout of an image, the tiling mode cannot be changed at a later time. If you want to be able to
        // directly access texels in the memory of the image, then you must use VK_IMAGE_TILING_LINEAR. We will be using a
        // staging buffer instead of a staging image, so this won't be necessary. We will be using VK_IMAGE_TILING_OPTIMAL
        // for efficient access from the shader.

        // There are only two possible values for the initialLayout of an image:
        //
        // VK_IMAGE_LAYOUT_UNDEFINED: Not usable by the GPU and the very first transition will discard the texels.
        // VK_IMAGE_LAYOUT_PREINITIALIZED: Not usable by the GPU, but the first transition will preserve the texels.
        //
        // There are few situations where it is necessary for the texels to be preserved during the first transition. One
        // example, however, would be if you wanted to use an image as a staging image in combination with the
        // VK_IMAGE_TILING_LINEAR layout. In that case, you'd want to upload the texel data to it and then transition the
        // image to be a transfer source without losing the data. In our case, however, we're first going to transition the
        // image to be a transfer destination and then copy texel data to it from a buffer object, so we don't need this
        // property and can safely use VK_IMAGE_LAYOUT_UNDEFINED.

        // The samples flag is related to multisampling. This is only relevant for images that will be used as attachments,
        // so stick to one sample.

        // There are some optional flags for images that are related to sparse images. Sparse images
        // are images where only certain regions are actually backed by memory. If you were using a 3D texture for a voxel
        // terrain, for example, then you could use this to avoid allocating memory to store large volumes of "air" values.
        // We won't be using it in this tutorial, so leave it to its default value of 0.
        imageInfo.flags = 0;  // Optional

        auto result = vkCreateImage(device, &imageInfo, nullptr, &image);
        throwOnError(result, "failed to create image!");

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, image, &memRequirements);

        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType                = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize       = memRequirements.size;
        allocInfo.memoryTypeIndex      = findMemoryType(memRequirements.memoryTypeBits, properties);

        result = vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory);
        throwOnError(result, "failed to allocate image memory!");

        vkBindImageMemory(device, image, imageMemory, 0);
    }

    void createDescriptorSets()
    {
        std::vector<VkDescriptorSetLayout> layouts(swapChainImages.size(), descriptorSetLayout);
        VkDescriptorSetAllocateInfo        allocInfo = {};
        allocInfo.sType                              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool                     = descriptorPool;
        allocInfo.descriptorSetCount                 = static_cast<uint32_t>(swapChainImages.size());
        allocInfo.pSetLayouts                        = layouts.data();

        descriptorSets.resize(swapChainImages.size());

        auto result = vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data());
        throwOnError(result, "failed to allocate descriptor sets!");

        for (size_t i = 0; i < swapChainImages.size(); i++)
        {
            VkDescriptorBufferInfo bufferInfo = {};
            bufferInfo.buffer                 = uniformBuffers[i];
            bufferInfo.offset                 = 0;
            bufferInfo.range                  = sizeof(UniformBufferObject);  // VK_WHOLE_SIZE

            VkWriteDescriptorSet descriptorWrite = {};
            descriptorWrite.sType                = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrite.dstSet               = descriptorSets[i];
            descriptorWrite.dstBinding           = 0;
            descriptorWrite.dstArrayElement      = 0;

            descriptorWrite.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrite.descriptorCount = 1;

            descriptorWrite.pBufferInfo      = &bufferInfo;
            descriptorWrite.pImageInfo       = nullptr;  // Optional
            descriptorWrite.pTexelBufferView = nullptr;  // Optional

            vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
        }
    }

    void createDescriptorPool()
    {
        VkDescriptorPoolSize poolSize = {};
        poolSize.type                 = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSize.descriptorCount      = static_cast<uint32_t>(swapChainImages.size());

        VkDescriptorPoolCreateInfo poolInfo = {};
        poolInfo.sType                      = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount              = 1;
        poolInfo.pPoolSizes                 = &poolSize;

        poolInfo.maxSets = static_cast<uint32_t>(swapChainImages.size());

        // The structure has an optional flag similar to command pools that determines if individual descriptor sets can be
        // freed or not: VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT. We're not going to touch the descriptor set after
        // creating it, so we don't need this flag. You can leave flags to its default value of 0.

        auto result = vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool);
        throwOnError(result, "failed to create descriptor pool!");
    }

    void createUniformBuffers()
    {
        VkDeviceSize bufferSize = sizeof(UniformBufferObject);

        uniformBuffers.resize(swapChainImages.size());
        uniformBuffersMemory.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++)
        {
            createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i],
                         uniformBuffersMemory[i]);
        }
    }

    void createDescriptorSetLayout()
    {
        VkDescriptorSetLayoutBinding uboLayoutBinding = {};
        uboLayoutBinding.binding                      = 0;
        uboLayoutBinding.descriptorType               = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboLayoutBinding.descriptorCount              = 1;

        // VK_SHADER_STAGE_ALL_GRAPHICS (all of the below)
        // VK_SHADER_STAGE_VERTEX_BIT,
        // VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
        // VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
        // VK_SHADER_STAGE_GEOMETRY_BIT,
        // VK_SHADER_STAGE_FRAGMENT_BIT,
        // VK_SHADER_STAGE_COMPUTE_BIT,
        // VK_SHADER_STAGE_ALL_GRAPHICS,
        // VK_SHADER_STAGE_ALL,
        // VK_SHADER_STAGE_RAYGEN_BIT_NV,
        // VK_SHADER_STAGE_ANY_HIT_BIT_NV,
        // VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV,
        // VK_SHADER_STAGE_MISS_BIT_NV,
        // VK_SHADER_STAGE_INTERSECTION_BIT_NV,
        // VK_SHADER_STAGE_CALLABLE_BIT_NV,
        // VK_SHADER_STAGE_TASK_BIT_NV,
        // VK_SHADER_STAGE_MESH_BIT_NV,
        // VK_SHADER_STAGE_FLAG_BITS_MAX_ENUM
        uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        // pImmutableSamplers field is only relevant for image sampling related descriptors
        uboLayoutBinding.pImmutableSamplers = nullptr;  // Optional

        VkDescriptorSetLayoutCreateInfo layoutInfo = {};
        layoutInfo.sType                           = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount                    = 1;
        layoutInfo.pBindings                       = &uboLayoutBinding;

        auto result = vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout);
        throwOnError(result, "failed to create descriptor set layout!");
    }

    void createIndexBuffer()
    {
        VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

        VkBuffer       stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer,
                     stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, indices.data(), (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

        copyBuffer(stagingBuffer, indexBuffer, bufferSize);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    void createVertexBuffer()
    {
        VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

        VkBuffer       stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer,
                     stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, vertices.data(), (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);

        copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);

        // You can now simply memcpy the vertex data to the mapped memory and unmap it again using vkUnmapMemory.
        // Unfortunately the driver may not immediately copy the data into the buffer memory, for example because of caching.
        // It is also possible that writes to the buffer are not visible in the mapped memory yet. There are two ways to deal
        // with that problem:
        //
        // Use a memory heap that is host coherent, indicated with VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        //
        // Call vkFlushMappedMemoryRanges to after writing to the mapped memory, and call vkInvalidateMappedMemoryRanges
        // before reading from the mapped memory

        // The transfer of data to the GPU is an operation that happens in the background and the specification simply tells
        // us that it is guaranteed to be complete as of the next call to vkQueueSubmit.
    }

    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout)
    {
        // One of the most common ways to perform layout transitions is using an image memory barrier. A pipeline barrier
        // like that is generally used to synchronize access to resources, like ensuring that a write to a buffer completes
        // before reading from it, but it can also be used to transition image layouts and transfer queue family ownership
        // when VK_SHARING_MODE_EXCLUSIVE is used. There is an equivalent buffer memory barrier to do this for buffers.
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkImageMemoryBarrier barrier = {};
        barrier.sType                = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout            = oldLayout;
        barrier.newLayout            = newLayout;

        // If you are using the barrier to transfer queue family ownership, then these two fields should be the indices of
        // the queue families. They must be set to VK_QUEUE_FAMILY_IGNORED if you don't want to do this (not the default
        // value!).
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        barrier.image                           = image;
        barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel   = 0;
        barrier.subresourceRange.levelCount     = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount     = 1;

        barrier.srcAccessMask = 0;  // TODO
        barrier.dstAccessMask = 0;  // TODO

        // All types of pipeline barriers are submitted using the same function. The first parameter after the command buffer
        // specifies in which pipeline stage the operations occur that should happen before the barrier. The second parameter
        // specifies the pipeline stage in which operations will wait on the barrier. The pipeline stages that you are allowed
        // to specify before and after the barrier depend on how you use the resource before and after the barrier. The
        // allowed values are listed in this table of the specification. For example, if you're going to read from a uniform
        // after the barrier, you would specify a usage of VK_ACCESS_UNIFORM_READ_BIT and the earliest shader that will read
        // from the uniform as pipeline stage, for example VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT. It would not make sense to
        // specify a non-shader pipeline stage for this type of usage and the validation layers will warn you when you specify
        // a pipeline stage that does not match the type of usage.
        //
        // The third parameter is either 0 or VK_DEPENDENCY_BY_REGION_BIT. The latter turns the barrier into a per-region
        // condition. That means that the implementation is allowed to already begin reading from the parts of a resource that
        // were written so far, for example.
        //
        // The last three pairs of parameters reference arrays of pipeline barriers of the three available types: memory
        // barriers, buffer memory barriers, and image memory barriers like the one we're using here. Note that we're not
        // using the VkFormat parameter yet, but we'll be using that one for special transitions in the depth buffer chapter.
        vkCmdPipelineBarrier(commandBuffer, 0 /* TODO */, 0 /* TODO */, 0, 0, nullptr, 0, nullptr, 1, &barrier);

        endSingleTimeCommands(commandBuffer);
    }

	void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height)
    {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkBufferImageCopy region = {};
        region.bufferOffset      = 0;
        region.bufferRowLength   = 0;
        region.bufferImageHeight = 0;

        region.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel       = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount     = 1;

        region.imageOffset = {0, 0, 0};
        region.imageExtent = {width, height, 1};

		vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        endSingleTimeCommands(commandBuffer);
    }

    // TODO: @MaxCompleteAPI, maybe rename this to buffercpy, also reorder params to (dst, src, size) (same as memcpy)
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
    {
        // TODO: @MaxCompleteAPI;
        // Memory transfer operations are executed using command buffers, just like drawing commands. Therefore we must first
        // allocate a temporary command buffer. You may wish to create a separate command pool for these kinds of short-lived
        // buffers, because the implementation may be able to apply memory allocation optimizations. You should use the
        // VK_COMMAND_POOL_CREATE_TRANSIENT_BIT flag during command pool generation in that case.

        // TODO: @MaxCompleteAPI;
        // There are again two possible ways to wait on this transfer to complete. We could use a fence and wait with
        // vkWaitForFences, or simply wait for the transfer queue to become idle with vkQueueWaitIdle. A fence would allow
        // you to schedule multiple transfers simultaneously and wait for all of them complete, instead of executing one at a
        // time. That may give the driver more opportunities to optimize.

        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkBufferCopy copyRegion = {};
        copyRegion.size         = size;
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

        endSingleTimeCommands(commandBuffer);
    }

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer,
                      VkDeviceMemory& bufferMemory)
    {
        VkBufferCreateInfo bufferInfo = {};
        bufferInfo.sType              = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size               = size;

        // The second field is usage, which indicates for which purposes the data in the buffer is going to be used. It is
        // possible to specify multiple purposes using a bitwise or.
        bufferInfo.usage = usage;

        // Just like the images in the swap chain, buffers can also be owned by a specific queue family or be shared between
        // multiple at the same time. The buffer will only be used from the graphics queue, so we can stick to exclusive
        // access.
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create buffer!");
        }

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType                = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize       = memRequirements.size;
        allocInfo.memoryTypeIndex      = findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to allocate buffer memory!");
        }

        // fourth parameter is the offset within the region of memory. Since this memory is allocated specifically for this
        // the vertex buffer, the offset is simply 0. If the offset is non-zero, then it is required to be divisible by
        // memRequirements.alignment.
        vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
    {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        // The VkPhysicalDeviceMemoryProperties structure has two arrays memoryTypes and memoryHeaps. Memory heaps are
        // distinct memory resources like dedicated VRAM and swap space in RAM for when VRAM runs out. The different types of
        // memory exist within these heaps.
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
        {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
            {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    void recreateSwapChain()
    {
        // TODO: @MaxYikes, fix this shit with semaphores or something please
        int width = 0, height = 0;
        while (width == 0 || height == 0)
        {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(device);

        cleanupSwapChain();

        createSwapChain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();
        createUniformBuffers();
        createDescriptorPool();
        createCommandBuffers();
    }

    void createSyncObjects()
    {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        VkSemaphoreCreateInfo semaphoreInfo = {};
        semaphoreInfo.sType                 = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo = {};
        fenceInfo.sType             = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags             = VK_FENCE_CREATE_SIGNALED_BIT;

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            auto result = vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]);
            throwOnError(result, "failed to create image-semaphore!");

            result = vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]);
            throwOnError(result, "failed to create finished-semaphore!");

            result = vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]);
            throwOnError(result, "failed to create fence!");
        }
    }

    void createCommandBuffers()
    {
        commandBuffers.resize(swapChainFramebuffers.size());

        VkCommandBufferAllocateInfo allocInfo = {};
        allocInfo.sType                       = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool                 = commandPool;

        // The level parameter specifies if the allocated command buffers are primary or secondary command buffers.
        // VK_COMMAND_BUFFER_LEVEL_PRIMARY:   // Can be submitted to a queue for execution, but cannot be called from
        // other
        //									  // command buffers.
        // VK_COMMAND_BUFFER_LEVEL_SECONDARY: // Cannot be submitted directly, but can be called from primary command
        //									  // buffers.
        allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

        auto result = vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data());
        throwOnError(result, "failed to allocate command buffers!");

        for (size_t i = 0; i < commandBuffers.size(); i++)
        {
            VkCommandBufferBeginInfo beginInfo = {};
            beginInfo.sType                    = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

            // The flags parameter specifies how we're going to use the command buffer. The following values are
            // available:

            // VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT:		 // The command buffer will be rerecorded right after
            //													 // executing it once.
            // VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT: // This is a secondary command buffer that will be
            // entirely
            //													 // within a single render pass.
            // VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT:	 // The command buffer can be resubmitted while it is
            // also
            //													 // already pending execution.
            beginInfo.flags = 0;  // Optional

            // The pInheritanceInfo parameter is only relevant for secondary command buffers. It specifies which state to
            // inherit from the calling primary command buffers.
            beginInfo.pInheritanceInfo = nullptr;  // Optional

            auto result = vkBeginCommandBuffer(commandBuffers[i], &beginInfo);
            throwOnError(result, "failed to begin recording command buffer!");

            VkRenderPassBeginInfo renderPassInfo = {};
            renderPassInfo.sType                 = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            renderPassInfo.renderPass            = renderPass;
            renderPassInfo.framebuffer           = swapChainFramebuffers[i];

            renderPassInfo.renderArea.offset = {0, 0};
            renderPassInfo.renderArea.extent = swapChainExtent;

            VkClearValue clearColor        = {0.0f, 0.0f, 0.0f, 1.0f};
            renderPassInfo.clearValueCount = 1;
            renderPassInfo.pClearValues    = &clearColor;

            vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

            vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

            VkBuffer     vertexBuffers[] = {vertexBuffer};
            VkDeviceSize offsets[]       = {0};
            vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, vertexBuffers, offsets);

            // you can only have a single index buffer. It's unfortunately not possible to use different indices for each
            // vertex attribute, so we do still have to completely duplicate vertex data even if just one attribute varies.
            vkCmdBindIndexBuffer(commandBuffers[i], indexBuffer, 0, VK_INDEX_TYPE_UINT16);

            vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
                                    &descriptorSets[i], 0, nullptr);

            // OLD call before we indexed vertices
            // vkCmdDraw(commandBuffers[i], static_cast<uint32_t>(vertices.size()), 1, 0, 0);
            vkCmdDrawIndexed(commandBuffers[i], static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

            vkCmdEndRenderPass(commandBuffers[i]);

            result = vkEndCommandBuffer(commandBuffers[i]);
            throwOnError(result, "failed to record command buffer!");
        }
    }

    void createCommandPool()
    {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        VkCommandPoolCreateInfo poolInfo = {};
        poolInfo.sType                   = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex        = queueFamilyIndices.graphicsFamily.value();

        // There are two possible flags for command pools:

        // VK_COMMAND_POOL_CREATE_TRANSIENT_BIT:			// Hint that command buffers are rerecorded with new commands
        //													// very often (may change memory allocation behavior)
        // VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT: // Allow command buffers to be rerecorded individually,
        // without
        //													// this flag they all have to be reset together
        poolInfo.flags = 0;  // Optional

        auto result = vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool);
        throwOnError(result, "failed to create command pool!");
    }

    void createFramebuffers()
    {
        swapChainFramebuffers.resize(swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); i++)
        {
            VkImageView attachments[] = {swapChainImageViews[i]};

            VkFramebufferCreateInfo framebufferInfo = {};
            framebufferInfo.sType                   = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass              = renderPass;
            framebufferInfo.attachmentCount         = 1;
            framebufferInfo.pAttachments            = attachments;
            framebufferInfo.width                   = swapChainExtent.width;
            framebufferInfo.height                  = swapChainExtent.height;
            framebufferInfo.layers = 1;  // layers refers to the number of layers in image arrays. Our swap chain images
                                         // are single images, so the number of layers is 1.

            auto result = vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]);
            throwOnError(result, "failed to create framebuffer!");
        }
    }

    void createRenderPass()
    {
        VkAttachmentDescription colorAttachment = {};
        colorAttachment.format                  = swapChainImageFormat;
        colorAttachment.samples                 = VK_SAMPLE_COUNT_1_BIT;  // multisampling

        // The loadOp and storeOp apply to color and depth data

        // We have the following choices for loadOp:
        // VK_ATTACHMENT_LOAD_OP_LOAD:		// Preserve the existing contents of the attachment
        // VK_ATTACHMENT_LOAD_OP_CLEAR:		// Clear the values to a constant at the start
        // VK_ATTACHMENT_LOAD_OP_DONT_CARE: // Existing contents are undefined; we don't care about them
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;

        // There are only two possibilities for the storeOp:
        // VK_ATTACHMENT_STORE_OP_STORE:	 // Rendered contents will be stored in memory and can be read later
        // VK_ATTACHMENT_STORE_OP_DONT_CARE: // Contents of the framebuffer will be undefined after the rendering
        // operation
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        colorAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout   = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        // Some of the most common layouts are:
        // VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL: // Images used as color attachment
        // VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:			 // Images to be presented in the swap chain
        // VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:	 // Images to be used as destination for a memory copy operation

        // Subpasses are subsequent rendering operations that depend on the contents of framebuffers in previous passes,
        // for example a sequence of post-processing effects that are applied one after another. If you group these
        // rendering operations into one render pass, then Vulkan is able to reorder the operations and conserve memory
        // bandwidth for possibly better performance.

        VkAttachmentReference colorAttachmentRef = {};
        colorAttachmentRef.attachment            = 0;
        colorAttachmentRef.layout                = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass = {};
        // Vulkan may also support compute subpasses in the future, so we have to be explicit about this being a graphics
        // subpass.
        subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments    = &colorAttachmentRef;

        // The following other types of attachments can be referenced by a subpass:
        // pInputAttachments:		// Attachments that are read from a shader
        // pResolveAttachments:		// Attachments used for multisampling color attachments
        // pDepthStencilAttachment: // Attachment for depth and stencil data
        // pPreserveAttachments:	// Attachments that are not used by this subpass, but for which the data must be
        // preserved

        VkRenderPassCreateInfo renderPassInfo = {};
        renderPassInfo.sType                  = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount        = 1;
        renderPassInfo.pAttachments           = &colorAttachment;
        renderPassInfo.subpassCount           = 1;
        renderPassInfo.pSubpasses             = &subpass;

        // There are two built-in dependencies that take care of the transition at the start of the render pass and at
        // the end of the render pass, but the former does not occur at the right time. It assumes that the transition
        // occurs at the start of the pipeline, but we haven't acquired the image yet at that point! There are two ways
        // to deal with this problem. We could change the waitStages for the imageAvailableSemaphore to
        // VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT to ensure that the render passes don't begin until the image is available,
        // or we can make the render pass wait for the VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT stage.

        VkSubpassDependency dependency = {};
        dependency.srcSubpass          = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass          = 0;

        // The next two fields specify the operations to wait on and the stages in which these operations occur. We need
        // to wait for the swap chain to finish reading from the image before we can access it. This can be accomplished
        // by waiting on the color attachment output stage itself.
        dependency.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;

        dependency.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies   = &dependency;

        auto result = vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass);
        throwOnError(result, "failed to create render pass!");
    }

    void createGraphicsPipeline()
    {
        auto vertShaderCode = readFile("shaders/vert.spv");
        auto fragShaderCode = readFile("shaders/frag.spv");

        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
        vertShaderStageInfo.sType                           = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage  = VK_SHADER_STAGE_VERTEX_BIT;  // There is an enum value for all programmable stages
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName  = "main";  // the function to invoke, known as the entrypoint. That means that it's
                                              // possible to combine multiple fragment shaders into a single shader module
                                              // and use different entry points to differentiate between their behaviors

        // There is one more (optional) member, pSpecializationInfo, which we won't be using here, but is worth
        // discussing. It allows you to specify values for shader constants. You can use a single shader module where its
        // behavior can be configured at pipeline creation by specifying different values for the constants used in it.
        // This is more efficient than configuring the shader using variables at render time, because the compiler can do
        // optimizations like eliminating if statements that depend on these values. If you don't have any constants like
        // that, then you can set the member to nullptr, which our struct initialization does automatically.
        vertShaderStageInfo.pSpecializationInfo = nullptr;

        VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
        fragShaderStageInfo.sType                           = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage                           = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module                          = fragShaderModule;
        fragShaderStageInfo.pName                           = "main";

        std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages = {vertShaderStageInfo, fragShaderStageInfo};

        // TODO: @MaxCompleteAPI, hardcoding the Vertex data as the accepted for this pipeline, make this accept any object
        // and be dynamic (TEMPLATES?!?!?! WOOOOHOO)
        auto bindingDescription    = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        // Bindings: // spacing between data and whether the data is per-vertex or per-instance
        //           // https://en.wikipedia.org/wiki/Geometry_instancing
        // Attribute descriptions: // type of the attributes passed to the vertex shader,which binding to load them from and
        //                         // at which offset
        VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
        vertexInputInfo.sType                                = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount        = 1;
        vertexInputInfo.vertexAttributeDescriptionCount      = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions           = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions         = attributeDescriptions.data();

        // The VkPipelineInputAssemblyStateCreateInfo struct describes two things: what kind of geometry will be drawn
        // from the vertices and if primitive restart should be enabled. The former is specified in the topology member
        // and can have values like:
        //
        // VK_PRIMITIVE_TOPOLOGY_POINT_LIST:	 points from vertices
        // VK_PRIMITIVE_TOPOLOGY_LINE_LIST:		 line from every 2 vertices without reuse
        // VK_PRIMITIVE_TOPOLOGY_LINE_STRIP:	 the end vertex of every line is used as start vertex for the next line
        // VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:  triangle from every 3 vertices without reuse
        // VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP: the second and third vertex of every triangle are used as first two
        // vertices
        //										 of the next triangle
        //
        // Normally, the vertices are loaded from the vertex buffer by index in sequential order, but
        // with an element buffer you can specify the indices to use yourself. This allows you to perform optimizations
        // like reusing vertices. If you set the primitiveRestartEnable member to VK_TRUE, then it's possible to break up
        // lines and triangles in the _STRIP topology modes by using a special index of 0xFFFF or 0xFFFFFFFF.

        // only triangles
        VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
        inputAssembly.sType                                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology                               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable                 = VK_FALSE;

        VkViewport viewport = {};
        viewport.x          = 0.0f;
        viewport.y          = 0.0f;
        viewport.width      = (float)swapChainExtent.width;
        viewport.height     = (float)swapChainExtent.height;
        viewport.minDepth   = 0.0f;
        viewport.maxDepth   = 1.0f;
        // The minDepth and maxDepth values specify the range of depth values to use for the framebuffer. These values
        // must be within the [0.0f, 1.0f] range, but minDepth may be higher than maxDepth. If you aren't doing anything
        // special, then you should stick to the standard values of 0.0f and 1.0f.

        // scissor is a filter rectangle where anything outside of it will be discarded by the rasterizer
        VkRect2D scissor = {};
        scissor.offset   = {0, 0};
        scissor.extent   = swapChainExtent;

        //  It is possible to use multiple viewports and scissor rectangles on some graphics cards, so its members
        //  reference an array of them. Using multiple requires enabling a GPU feature
        // see ->
        // https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Logical_device_and_queues#page_Creating-the-logical-device).
        VkPipelineViewportStateCreateInfo viewportState = {};
        viewportState.sType                             = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount                     = 1;
        viewportState.pViewports                        = &viewport;
        viewportState.scissorCount                      = 1;
        viewportState.pScissors                         = &scissor;

        // The rasterizer takes the geometry that is shaped by the vertices from the vertex shader and turns it into
        // fragments to be colored by the fragment shader. It also performs depth testing, face culling and the scissor
        // test, and it can be configured to output fragments that fill entire polygons or just the edges (wireframe
        // rendering). All this is configured using the VkPipelineRasterizationStateCreateInfo structure.
        VkPipelineRasterizationStateCreateInfo rasterizer = {};
        rasterizer.sType                                  = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;

        // If depthClampEnable is set to VK_TRUE, then fragments that are beyond the near and far planes are clamped to
        // them as opposed to discarding them. This is useful in some special cases like shadow maps. Using this requires
        // enabling a GPU feature.
        rasterizer.depthClampEnable = VK_FALSE;

        // If rasterizerDiscardEnable is set to VK_TRUE, then geometry never passes through the rasterizer stage. This
        // basically disables any output to the framebuffer.
        rasterizer.rasterizerDiscardEnable = VK_FALSE;

        // The polygonMode determines how fragments are generated for geometry. The following modes are available:

        // VK_POLYGON_MODE_FILL:  // fill the area of the polygon with fragments
        // VK_POLYGON_MODE_LINE:  // polygon edges are drawn as lines
        // VK_POLYGON_MODE_POINT: // polygon vertices are drawn as points Using any mode other than fill requires
        // enabling a
        //						  // GPU feature.
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;

        // lineWidth describes the thickness of lines in terms of number of fragments. The maximum line width that is
        // supported depends on the hardware and any line thicker than 1.0f requires you to enable the wideLines GPU
        // feature.
        rasterizer.lineWidth = 1.0f;

        // The cullMode variable determines the type of face culling to use. You can disable culling, cull the front
        // faces, cull the back faces or both.
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;

        // The frontFace variable specifies the vertex order for faces to be considered front-facing and can be clockwise
        // or counterclockwise.
        // This is counterclockwise because we need to flip everything because of GLM being based on OpenGL and they have the
        // y axis flipped
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

        // The rasterizer can alter the depth values by adding a constant value or biasing them based on a fragment's
        // slope. This is sometimes used for shadow mapping.
        rasterizer.depthBiasEnable         = VK_FALSE;
        rasterizer.depthBiasConstantFactor = 0.0f;  // Optional
        rasterizer.depthBiasClamp          = 0.0f;  // Optional
        rasterizer.depthBiasSlopeFactor    = 0.0f;  // Optional

        // The VkPipelineMultisampleStateCreateInfo struct configures multisampling, which is one of the ways to perform
        // anti-aliasing. It works by combining the fragment shader results of multiple polygons that rasterize to the
        // same pixel. This mainly occurs along edges, which is also where the most noticeable aliasing artifacts occur.
        // Because it doesn't need to run the fragment shader multiple times if only one polygon maps to a pixel, it is
        // significantly less expensive than simply rendering to a higher resolution and then downscaling. Enabling it
        // requires enabling a GPU feature.
        VkPipelineMultisampleStateCreateInfo multisampling = {};
        multisampling.sType                                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable                  = VK_FALSE;
        multisampling.rasterizationSamples                 = VK_SAMPLE_COUNT_1_BIT;
        multisampling.minSampleShading                     = 1.0f;      // Optional
        multisampling.pSampleMask                          = nullptr;   // Optional
        multisampling.alphaToCoverageEnable                = VK_FALSE;  // Optional
        multisampling.alphaToOneEnable                     = VK_FALSE;  // Optional

        // If you are using a depth and/or stencil buffer, then you also need to configure the depth and stencil tests
        // using VkPipelineDepthStencilStateCreateInfo. We don't have one right now, so we can simply pass a nullptr
        // instead of a pointer to such a struct. We'll get back to it in the depth buffering chapter.

        // The first struct, VkPipelineColorBlendAttachmentState contains the configuration per attached framebuffer and
        // the second struct, VkPipelineColorBlendStateCreateInfo contains the global color blending settings. In our
        // case we only have one framebuffer
        VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
        colorBlendAttachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        // yes blending
        colorBlendAttachment.blendEnable         = VK_TRUE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorBlendAttachment.colorBlendOp        = VK_BLEND_OP_ADD;
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment.alphaBlendOp        = VK_BLEND_OP_ADD;

        // no blending
        // colorBlendAttachment.blendEnable         = VK_FALSE;
        // colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;   // Optional
        // colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;  // Optional
        // colorBlendAttachment.colorBlendOp        = VK_BLEND_OP_ADD;       // Optional
        // colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;   // Optional
        // colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;  // Optional
        // colorBlendAttachment.alphaBlendOp        = VK_BLEND_OP_ADD;       // Optional

        VkPipelineColorBlendStateCreateInfo colorBlending = {};
        colorBlending.sType                               = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable                       = VK_FALSE;
        colorBlending.logicOp                             = VK_LOGIC_OP_COPY;  // Optional
        colorBlending.attachmentCount                     = 1;
        colorBlending.pAttachments                        = &colorBlendAttachment;
        colorBlending.blendConstants[0]                   = 0.0f;  // Optional
        colorBlending.blendConstants[1]                   = 0.0f;  // Optional
        colorBlending.blendConstants[2]                   = 0.0f;  // Optional
        colorBlending.blendConstants[3]                   = 0.0f;  // Optional

        // A limited amount of the state that we 've specified in the previous structs can actually be changed without
        // recreating the pipeline. Examples are the size of the viewport, line width and blend constants. If you want to
        // do that, then you'll have to fill in a VkPipelineDynamicStateCreateInfo structure like this : VkDynamicState
        // dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_LINE_WIDTH};

        // VkPipelineDynamicStateCreateInfo dynamicState = {};
        // dynamicState.sType                            = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        // dynamicState.dynamicStateCount                = 2;
        // dynamicState.pDynamicStates                   = dynamicStates;

        // The structure also specifies push constants, which are another way of passing dynamic values to shaders
        VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
        pipelineLayoutInfo.sType                      = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount             = 1;
        pipelineLayoutInfo.pSetLayouts                = &descriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount     = 0;        // Optional
        pipelineLayoutInfo.pPushConstantRanges        = nullptr;  // Optional

        auto result = vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout);
        throwOnError(result, "failed to create pipeline layout!");

        VkGraphicsPipelineCreateInfo pipelineInfo = {};
        pipelineInfo.sType                        = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount                   = 2;
        pipelineInfo.pStages                      = shaderStages.data();

        pipelineInfo.pVertexInputState   = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState      = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState   = &multisampling;
        pipelineInfo.pDepthStencilState  = nullptr;  // Optional
        pipelineInfo.pColorBlendState    = &colorBlending;
        pipelineInfo.pDynamicState       = nullptr;  // Optional

        pipelineInfo.layout = pipelineLayout;

        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass    = 0;

        // There are actually two more parameters: basePipelineHandle and basePipelineIndex. Vulkan allows you to create
        // a new graphics pipeline by deriving from an existing pipeline. The idea of pipeline derivatives is that it is
        // less expensive to set up pipelines when they have much functionality in common with an existing pipeline and
        // switching between pipelines from the same parent can also be done quicker. You can either specify the handle
        // of an existing pipeline with basePipelineHandle or reference another pipeline that is about to be created by
        // index with basePipelineIndex. Right now there is only a single pipeline, so we'll simply specify a null handle
        // and an invalid index. These values are only used if the VK_PIPELINE_CREATE_DERIVATIVE_BIT flag is also
        // specified in the flags field of VkGraphicsPipelineCreateInfo.
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;  // Optional
        pipelineInfo.basePipelineIndex  = -1;              // Optional

        // The vkCreateGraphicsPipelines function actually has more parameters than the usual object creation functions
        // in Vulkan. It is designed to take multiple VkGraphicsPipelineCreateInfo objects and create multiple VkPipeline
        // objects in a single call.
        //
        // The second parameter, for which we've passed the VK_NULL_HANDLE argument, references an optional
        // VkPipelineCache object. A pipeline cache can be used to store and reuse data relevant to pipeline creation
        // across multiple calls to vkCreateGraphicsPipelines and even across program executions if the cache is stored
        // to a file. This makes it possible to significantly speed up pipeline creation at a later time. We'll get into
        // this in the pipeline cache chapter.
        result = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline);
        throwOnError(result, "failed to create graphics pipeline!");

        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
    }

    VkCommandBuffer beginSingleTimeCommands()
    {
        VkCommandBufferAllocateInfo allocInfo = {};
        allocInfo.sType                       = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level                       = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool                 = commandPool;
        allocInfo.commandBufferCount          = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType                    = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags                    = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        return commandBuffer;
    }

    void endSingleTimeCommands(VkCommandBuffer commandBuffer)
    {
        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo       = {};
        submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers    = &commandBuffer;

        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);

        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }

    // Creating a shader module is simple, we only need to specify a pointer to the buffer with the bytecode and the
    // length of it. This information is specified in a VkShaderModuleCreateInfo structure. The one catch is that the
    // size of the bytecode is specified in bytes, but the bytecode pointer is a uint32_t pointer rather than a char
    // pointer. Therefore we will need to cast the pointer with reinterpret_cast as shown below. When you perform a cast
    // like this, you also need to ensure that the data satisfies the alignment requirements of uint32_t. Lucky for us,
    // the data is stored in an std::vector where the default allocator already ensures that the data satisfies the worst
    // case alignment requirements.
    VkShaderModule createShaderModule(const std::vector<char>& code)
    {
        VkShaderModuleCreateInfo createInfo = {};
        createInfo.sType                    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize                 = code.size();
        createInfo.pCode                    = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shaderModule;

        auto result = vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule);
        throwOnError(result, "failed to create shader module!");

        return shaderModule;
    }

    void createImageViews()
    {
        swapChainImageViews.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++)
        {
            VkImageViewCreateInfo createInfo = {};
            createInfo.sType                 = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.image                 = swapChainImages[i];

            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;  // The viewType parameter allows you to treat images as 1D
                                                          // textures, 2D textures, 3D textures and cube maps.
            createInfo.format = swapChainImageFormat;

            // The components field allows you to swizzle the color channels around. For example, you can map all of the
            // channels to the red channel for a monochrome texture. You can also map constant values of 0 and 1 to a
            // channel. In our case we'll stick to the default mapping.
            createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

            // The subresourceRange field describes what the image's purpose is and which part of the image should be
            // accessed
            // the following are color targets without any mipmapping levels or multiple layers.
            createInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
            createInfo.subresourceRange.baseMipLevel   = 0;
            createInfo.subresourceRange.levelCount     = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount     = 1;

            // If you were working on a stereographic 3D application, then you would create a swap chain with multiple
            // layers. You could then create multiple image views for each image representing the views for the left and
            // right eyes by accessing different layers.

            auto result = vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]);
            throwOnError(result, "failed to create image views!");
        }
    }

    void createSwapChain()
    {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR   presentMode   = chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D         extent        = chooseSwapExtent(swapChainSupport.capabilities);

        // simply sticking to this minimum means that we may sometimes have to wait on the driver to complete internal
        // operations before we can acquire another image to render to. Therefore it is recommended to request at least
        // one more image than the minimum
        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        // 0 is a special value that means that there is no maximum
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
        {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        VkSwapchainCreateInfoKHR createInfo = {};
        createInfo.sType                    = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface                  = surface;

        createInfo.minImageCount    = imageCount;
        createInfo.imageFormat      = surfaceFormat.format;
        createInfo.imageColorSpace  = surfaceFormat.colorSpace;
        createInfo.imageExtent      = extent;
        createInfo.imageArrayLayers = 1;  // This is always 1 unless you are developing a stereoscopic 3D application
        createInfo.imageUsage =
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;  // It is also possible that you'll render images to a separate image
                                                  // first to perform operations like post-processing. In that case you
                                                  // may use a value like VK_IMAGE_USAGE_TRANSFER_DST_BIT instead and use
                                                  // a memory operation to transfer the rendered image to a swap chain
                                                  // image.

        QueueFamilyIndices indices              = findQueueFamilies(physicalDevice);
        uint32_t           queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};

        /*
        There are two ways to handle images that are accessed from multiple queues:
        VK_SHARING_MODE_EXCLUSIVE:
                An image is owned by one queue family at a time and ownership must be explicitly transfered before using
        it in another queue family.This option offers the best performance. VK_SHARING_MODE_CONCURRENT: Images can be
        used across multiple queue families without explicit ownership transfers.
        */
        if (indices.graphicsFamily != indices.presentFamily)
        {
            createInfo.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices   = queueFamilyIndices;
        }
        else
        {
            createInfo.imageSharingMode      = VK_SHARING_MODE_EXCLUSIVE;
            createInfo.queueFamilyIndexCount = 0;        // Optional
            createInfo.pQueueFamilyIndices   = nullptr;  // Optional
        }

        // We can specify that a certain transform should be applied to images in the swap chain if it is supported
        // (supportedTransforms in capabilities), like a 90 degree clockwise rotation or horizontal flip. To specify that
        // you do not want any transformation, simply specify the current transformation.
        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;

        // The compositeAlpha field specifies if the alpha channel should be used for blending with other windows in the
        // window system. You'll almost always want to simply ignore the alpha channel, hence
        // VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR.
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

        createInfo.presentMode = presentMode;

        // If the clipped member is set to VK_TRUE then that means that we don't care about the color of pixels that are
        // obscured, for example because another window is in front of them. Unless you really need to be able to read
        // these pixels back and get predictable results, you'll get the best performance by enabling clipping.
        createInfo.clipped = VK_TRUE;

        // That leaves one last field, oldSwapChain. With Vulkan it's possible that your swap chain becomes invalid or
        // unoptimized while your application is running, for example because the window was resized. In that case the
        // swap chain actually needs to be recreated from scratch and a reference to the old one must be specified in
        // this field. This is a complex topic that we'll learn more about in a future chapter. For now we'll assume that
        // we'll only ever create one swap chain.
        createInfo.oldSwapchain = VK_NULL_HANDLE;

        auto result = vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain);
        throwOnError(result, "failed to create swap chain!");

        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent      = extent;
    }

    void createSurface()
    {
        auto result = glfwCreateWindowSurface(instance, window, nullptr, &surface);
        throwOnError(result, "failed to create window surface!");
    }

    void createLogicalDevice()
    {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value()};

        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies)
        {
            VkDeviceQueueCreateInfo queueCreateInfo = {};
            queueCreateInfo.sType                   = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex        = queueFamily;
            queueCreateInfo.queueCount              = 1;
            queueCreateInfo.pQueuePriorities        = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        VkPhysicalDeviceFeatures deviceFeatures = {};

        VkDeviceCreateInfo createInfo = {};
        createInfo.sType              = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos    = queueCreateInfos.data();

        createInfo.pEnabledFeatures = &deviceFeatures;

        createInfo.enabledExtensionCount   = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        if (enableValidationLayers)
        {
            createInfo.enabledLayerCount   = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else
        {
            createInfo.enabledLayerCount = 0;
        }

        auto result = vkCreateDevice(physicalDevice, &createInfo, nullptr, &device);
        throwOnError(result, "failed to create logical device!");

        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
    }

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
    {
        for (const auto& availableFormat : availableFormats)
        {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM &&
                availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            {
                return availableFormat;
            }
        }

        // TODO: @MaxCompleteAPI, rank the available formats

        return availableFormats[0];
    }

    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes)
    {
        for (const auto& availablePresentMode : availablePresentModes)
        {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
            {
                return availablePresentMode;
            }
        }

        return VK_PRESENT_MODE_FIFO_KHR;

        /*
        VK_PRESENT_MODE_IMMEDIATE_KHR: Images submitted by your application are transferred to the screen right away,
        which may result in tearing. VK_PRESENT_MODE_FIFO_KHR: The swap chain is a queue where the display takes an image
        from the front of the queue when the display is refreshed and the program inserts rendered images at the back of
        the queue. If the queue is full then the program has to wait. This is most similar to vertical sync as found in
        modern games. The moment that the display is refreshed is known as "vertical blank".
        VK_PRESENT_MODE_FIFO_RELAXED_KHR: This mode only differs from the previous one if the application is late and the
        queue was empty at the last vertical blank. Instead of waiting for the next vertical blank, the image is
        transferred right away when it finally arrives. This may result in visible tearing. VK_PRESENT_MODE_MAILBOX_KHR:
        This is another variation of the second mode. Instead of blocking the application when the queue is full, the
        images that are already queued are simply replaced with the newer ones. This mode can be used to implement triple
        buffering, which allows you to avoid tearing with significantly less latency issues than standard vertical sync
        that uses double buffering.
        */
    }

    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
    {
        if (capabilities.currentExtent.width != UINT32_MAX)
        {
            return capabilities.currentExtent;
        }
        else
        {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            VkExtent2D actualExtent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};

            actualExtent.width =
                std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
            actualExtent.height = std::max(capabilities.minImageExtent.height,
                                           std::min(capabilities.maxImageExtent.height, actualExtent.height));

            return actualExtent;
        }
    }

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device)
    {
        SwapChainSupportDetails details;

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

        if (formatCount != 0)
        {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }

        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

        if (presentModeCount != 0)
        {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        }

        return details;
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

            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

            if (queueFamily.queueCount > 0 && presentSupport)
            {
                indices.presentFamily = i;
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

        bool extensionsSupported = checkDeviceExtensionSupport(device);
        if (!extensionsSupported) return 0;

        bool swapChainAdequate = false;
        if (extensionsSupported)
        {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }
        if (!swapChainAdequate) return 0;

        if (indices.graphicsFamily.value() == indices.presentFamily.value()) score += 10000;

        // Maximum possible size of textures affects graphics quality
        // score += deviceProperties.limits.maxImageDimension2D;

        // Application can't function without geometry shaders
        // if (!deviceFeatures.geometryShader)
        //{
        //    return 0;
        //}

        return score;
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice device)
    {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for (const auto& extension : availableExtensions)
        {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
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
            graphicsLog.Log("TIMESTAMP [ERROR]: " + message);
            throw std::runtime_error(message);
        }
    }

    void mainLoop()
    {
        while (!glfwWindowShouldClose(window))
        {
            glfwPollEvents();
            drawFrame();
        }

        vkDeviceWaitIdle(device);
    }

    void drawFrame()
    {
        // TODO: @MaxBestPractice, set the timeout
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
        vkResetFences(device, 1, &inFlightFences[currentFrame]);

        uint32_t imageIndex;
        // TODO: @MaxBestPractice, set the timeout
        auto result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame],
                                            VK_NULL_HANDLE, &imageIndex);

        // VK_SUBOPTIMAL_KHR is technically a success, and we have the image already but its not optimal
        // TODO: @MaxMaybe, Maybe do something special with VK_SUBOPTIMAL_KHR like drawing all the way then recreating the
        // SwapChain at the end of/after drawing the frame to completion
        if (result == VK_ERROR_OUT_OF_DATE_KHR || framebufferResized)
        {
            framebufferResized = false;
            recreateSwapChain();
            return;
        }
        else if (result == VK_SUBOPTIMAL_KHR)
        {
            framebufferResized = true;
        }
        else if (result != VK_SUCCESS)
        {
            throwOnError(result, "failed to acquire swap chain image!");
        }

        updateUniformBuffer(imageIndex);

        VkSubmitInfo submitInfo = {};
        submitInfo.sType        = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore          waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
        VkPipelineStageFlags waitStages[]     = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        submitInfo.waitSemaphoreCount         = 1;
        submitInfo.pWaitSemaphores            = waitSemaphores;
        submitInfo.pWaitDstStageMask          = waitStages;

        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers    = &commandBuffers[imageIndex];

        VkSemaphore signalSemaphores[]  = {renderFinishedSemaphores[currentFrame]};
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores    = signalSemaphores;

        vkResetFences(device, 1, &inFlightFences[currentFrame]);

        // The function takes an array of VkSubmitInfo structures as argument for efficiency when the workload is much
        // larger. The last parameter references an optional fence that will be signaled when the command buffers finish
        // execution.
        result = vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]);
        throwOnError(result, "failed to submit draw command buffer!");

        VkPresentInfoKHR presentInfo = {};
        presentInfo.sType            = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores    = signalSemaphores;

        VkSwapchainKHR swapChains[] = {swapChain};
        presentInfo.swapchainCount  = 1;
        presentInfo.pSwapchains     = swapChains;
        presentInfo.pImageIndices   = &imageIndex;

        // There is one last optional parameter called pResults. It allows you to specify an array of VkResult values to
        // check for every individual swap chain if presentation was successful. It's not necessary if you're only using
        // a single swap chain, because you can simply use the return value of the present function.
        presentInfo.pResults = nullptr;  // Optional

        result = vkQueuePresentKHR(presentQueue, &presentInfo);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized)
        {
            framebufferResized = false;
            recreateSwapChain();
        }
        else if (result != VK_SUCCESS)
        {
            throwOnError(result, "failed to present swap chain image!");
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    // TODO: @MaxCompleteAPI, Using a UBO this way is not the most efficient way to pass frequently changing values to the
    // shader. A more efficient way to pass a small buffer of data to shaders are push constants.
    void updateUniformBuffer(uint32_t currentImage)
    {
        static auto startTime = glfwGetTime();

        auto  currentTime = glfwGetTime();
        float time        = (currentTime - startTime);

        UniformBufferObject ubo = {};
        ubo.model               = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));

        ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));

        ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 10.0f);

        // GLM was originally designed for OpenGL, where the Y coordinate of the clip coordinates is inverted. The easiest
        // way to compensate for that is to flip the sign on the scaling factor of the Y axis in the projection matrix. If
        // you don't do this, then the image will be rendered upside down.
        ubo.proj[1][1] *= -1;

        void* data;
        vkMapMemory(device, uniformBuffersMemory[currentImage], 0, sizeof(ubo), 0, &data);
        memcpy(data, &ubo, sizeof(ubo));
        vkUnmapMemory(device, uniformBuffersMemory[currentImage]);
    }

    void cleanupSwapChain()
    {
        for (size_t i = 0; i < swapChainFramebuffers.size(); i++)
        {
            vkDestroyFramebuffer(device, swapChainFramebuffers[i], nullptr);
        }

        vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());

        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyRenderPass(device, renderPass, nullptr);

        for (auto imageView : swapChainImageViews)
        {
            vkDestroyImageView(device, imageView, nullptr);
        }

        vkDestroySwapchainKHR(device, swapChain, nullptr);

        for (size_t i = 0; i < swapChainImages.size(); i++)
        {
            vkDestroyBuffer(device, uniformBuffers[i], nullptr);
            vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
        }

        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    }

    void cleanup()
    {
        cleanupSwapChain();

        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

        vkDestroyBuffer(device, indexBuffer, nullptr);
        vkFreeMemory(device, indexBufferMemory, nullptr);

        vkDestroyBuffer(device, vertexBuffer, nullptr);
        vkFreeMemory(device, vertexBufferMemory, nullptr);  // used by buffer, deleted after it

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);
        }

        vkDestroyCommandPool(device, commandPool, nullptr);

        vkDestroyDevice(device, nullptr);

        // Must be before instance
        vkDestroySurfaceKHR(instance, surface, nullptr);

        // Must be before instance
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
            initLog("A:\\Development\\Vulkan Project\\VulkanTest\\x64\\Debug");
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

// The input assembler collects the raw vertex data from the buffers you specify and may also use an index buffer to
// repeat certain elements without having to duplicate the vertex data itself.
//
// The vertex shader is run for every vertex and generally applies transformations to turn vertex positions from model
// space to screen space. It also passes per-vertex data down the pipeline.
//
// The tessellation shaders allow you to subdivide geometry based on certain rules to increase the mesh quality. This is
// often used to make surfaces like brick walls and staircases look less flat when they are nearby.
//
// The geometry shader is run on every primitive (triangle, line, point) and can discard it or output more primitives
// than came in. This is similar to the tessellation shader, but much more flexible. However, it is not used much in
// today's applications because the performance is not that good on most graphics cards except for Intel's integrated
// GPUs.
//
// The rasterization stage discretizes the primitives into fragments. These are the pixel elements that they fill on the
// framebuffer. Any fragments that fall outside the screen are discarded and the attributes outputted by the vertex
// shader are interpolated across the fragments, as shown in the figure. Usually the fragments that are behind other
// primitive fragments are also discarded here because of depth testing.
//
// The fragment shader is invoked for every fragment that survives and determines which framebuffer(s) the fragments are
// written to and with which color and depth values. It can do this using the interpolated data from the vertex shader,
// which can include things like texture coordinates and normals for lighting.
//
// The color blending stage applies operations to mix different fragments that map to the same pixel in the framebuffer.
// Fragments can simply overwrite each other, add up or be mixed based upon transparency.
//
// *************************************************************************************************************************
//
// The graphics pipeline in Vulkan is almost completely immutable, so you must recreate the pipeline from scratch if you
// want to change shaders, bind different framebuffers or change the blend function. The disadvantage is that you'll have
// to create a number of pipelines that represent all of the different combinations of states you want to use in your
// rendering operations. However, because all of the operations you'll be doing in the pipeline are known in advance, the
// driver can optimize for it much better.
//
//
//
// Some of the programmable stages are optional based on what you intend to do. For example, the tessellation and
// geometry stages can be disabled if you are just drawing simple geometry. If you are only interested in depth values
// then you can disable the fragment shader stage, which is useful for shadow map generation.
// https://en.wikipedia.org/wiki/Shadow_mapping
//
// *************************************************************************************************************************
//
// Also try running the compiler without any arguments to see what kinds of flags it supports. It can, for example, also
// output the bytecode into a human-readable format so you can see exactly what your shader is doing and any
// optimizations that have been applied at this stage.
//
// *************************************************************************************************************************
//
// The Vulkan SDK includes libshaderc, which is a library to compile GLSL code to SPIR-V from within your program.
// https://github.com/google/shaderc
//
// *************************************************************************************************************************
//
// TODO: @MaxUpgradeAPI, Look into custom allocators for all the objects we are creating through vulkan calls
//
// *************************************************************************************************************************
//
// Fences are mainly designed to synchronize your application itself with rendering operation, whereas semaphores are
// used to synchronize operations within or across command queues.
//
// *************************************************************************************************************************
//
// That's all it takes to recreate the swap chain! However, the disadvantage of this approach is that we need to stop all
// rendering before creating the new swap chain. It is possible to create a new swap chain while drawing commands on an image
// from the old swap chain are still in-flight. You need to pass the previous swap chain to the oldSwapChain field in the
// VkSwapchainCreateInfoKHR struct and destroy the old swap chain as soon as you've finished using it.
//
// *************************************************************************************************************************
//
// The memoryTypes array consists of VkMemoryType structs that specify the heap and properties of each type of memory. The
// properties define special features of the memory, like being able to map it so we can write to it from the CPU. This
// property is indicated with VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, but we also need to use the
// VK_MEMORY_PROPERTY_HOST_COHERENT_BIT property.
//
// *************************************************************************************************************************
// TODO: @MaxAdvancedAPI, make a dedicated transfer queue, this will probably increase efficiency? (probably look that up)
// The buffer copy command requires a queue family that supports transfer operations, which is indicated using
// VK_QUEUE_TRANSFER_BIT. The good news is that any queue family with VK_QUEUE_GRAPHICS_BIT or VK_QUEUE_COMPUTE_BIT
// capabilities already implicitly support VK_QUEUE_TRANSFER_BIT operations. The implementation is not required to explicitly
// list it in queueFlags in those cases. If you like a challenge, then you can still try to use a different queue family
// specifically for transfer operations. It will require you to make the following modifications to your program:
//
//     - Modify QueueFamilyIndices and findQueueFamilies to explicitly look for a queue family with the VK_QUEUE_TRANSFER
//			bit, but not the VK_QUEUE_GRAPHICS_BIT.
//     - Modify createLogicalDevice to request a handle to the transfer queue
//     - Create a second command pool for command buffers that are submitted on the transfer queue family
//     - Change the sharingMode of resources to be VK_SHARING_MODE_CONCURRENT and specify both the graphics and transfer
//			queue families
//     - Submit transfer commands like vkCmdCopyBuffer to the transfer queue instead of the graphics queue
//
// *************************************************************************************************************************
//
// It should be noted that in a real world application, you're not supposed to actually call vkAllocateMemory for every
// individual buffer. The maximum number of simultaneous memory allocations is limited by the maxMemoryAllocationCount
// physical device limit, which may be as low as 4096 even on high end hardware like an NVIDIA GTX 1080. The right way to
// allocate memory for a large number of objects at the same time is to create a custom allocator that splits up a single
// allocation among many different objects by using the offset parameters that we've seen in many functions.
//
// You can either implement such an allocator yourself, or use the
// VulkanMemoryAllocator->(https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) library provided by the GPUOpen
// initiative.
//
// *************************************************************************************************************************
// TODO: @MaxCompleteAPI, this is best practice and likely much better on performance (also cache friendly :) )
// The previous chapter already mentioned that you should allocate multiple resources like buffers from a single memory
// allocation, but in fact you should go a step further. Driver developers recommend
// (https://developer.nvidia.com/vulkan-memory-management) that you also store multiple buffers, like the vertex and index
// buffer, into a single VkBuffer and use offsets in commands like vkCmdBindVertexBuffers. The advantage is that your data is
// more cache friendly in that case, because it's closer together. It is even possible to reuse the same chunk of memory for
// multiple resources if they are not used during the same render operations, provided that their data is refreshed, of
// course. This is known as aliasing and some Vulkan functions have explicit flags to specify that you want to do this.
//
// *************************************************************************************************************************
//
// To push further the software performance, a programmer can define GLM_FORCE_INLINE before any inclusion of <glm/glm.hpp>
// to force the compiler to inline GLM code. #define GLM_FORCE_INLINE #include <glm/glm.hpp>
//
// *************************************************************************************************************************
//
// Vulkan expects the data in your structure to be aligned in memory in a specific way, for example:
//
// Scalars have to be aligned by N (= 4 bytes given 32 bit floats).
// A vec2 must be aligned by 2N (= 8 bytes)
// A vec3 or vec4 must be aligned by 4N (= 16 bytes)
// A nested structure must be aligned by the base alignment of its members rounded up to a multiple of 16.
// A mat4 matrix must have the same alignment as a vec4.
//
// You can find the full list of alignment requirements in the specification.
// (https://www.khronos.org/registry/vulkan/specs/1.1-extensions/html/chap14.html#interfaces-resources-layout)
//
// *************************************************************************************************************************
//
// As some of the structures and function calls hinted at, it is actually possible to bind multiple descriptor sets
// simultaneously. You need to specify a descriptor layout for each descriptor set when creating the pipeline layout. Shaders
// can then reference specific descriptor sets like this:
//
//			layout(set = 0, binding = 0) uniform UniformBufferObject { ... }
//
// You can use this feature to put descriptors that vary per-object and descriptors that are shared into separate descriptor
// sets. In that case you avoid rebinding most of the descriptors across draw calls which is potentially more efficient.
//
// *************************************************************************************************************************
//
// Well, I actually starting to think something is wrong in glm::lookAt.
// I was digging a bit deeper according to the whole LH/RH issue. As it turns out glm has a #define for this GLM_LEFT_HANDED
// or GLM_RIGHT_HANDED and it will use the matching functions lookAtRH/LH and perspectiveRH/LH � but they changed this to be
// default on GLM_LEFT_HANDED (I guess because of Vulkan). I always forget which one is used by Direct3D and which by OpenGL,
// so I was searching � most sources note that: LH is Direct3d/Vulkan and RH is OpenGL. But, wait ... that means it simply
// should have worked (without the CCW rendering and without inverting anything by hand), right?
//
// Something was bugging me too (if you do not correct anything) you get an awkward coordinate system with Y-Axis down � that
// is even RH ... what the hell?
//
// I think it should be: X-Axis:Left, Y-Axis:Up, Z-Axis: Into which is LH. With lookAt Up vector set to vec3(0.0f, 1.0f,
// 0.0f). (Well with Unreal I�m actually used to a Z-Up System � but that is Unreal.)
//
// After a bit trial and error I was looking into the implementation of glm::lookAtLH and figured out that if you swap the
// second cross product in: tvec3<t, p=""> const u(cross(f, s)); to u(cross(s, f)); everything works exactly like in a Y-Up
// LH System, with VK_FRONT_FACE_CLOCKWISE set. So, maybe this is just a bug in glm?
//
// COMMENT ->
// I used #define GLM_FORCE_LEFT_HANDED before including GLM
// then the front face can be left at VK_FRONT_FACE_CLOCKWISE
// The only thing that needs to be changed is the Z coordinate of the camera because in left handed systems the positive Z
// moves away from the viewer. So: glm::lookAt(glm::vec3(0.0f, 2.0f, -2.0f), glm::vec3(0.0f, 0.0f, 0.0f),
// glm::vec3(0.0f, 1.0f, 0.0f));
//
// *************************************************************************************************************************
//
// When performing any operation on images, you must make sure that they have the layout that is optimal for use in that
// operation. We've actually already seen some of these layouts when we specified the render pass:
//
// VK_IMAGE_LAYOUT_PRESENT_SRC_KHR: Optimal for presentation
// VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL: Optimal as attachment for writing colors from the fragment shader
// VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL: Optimal as source in a transfer operation, like vkCmdCopyImageToBuffer
// VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL: Optimal as destination in a transfer operation, like vkCmdCopyBufferToImage
// VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL: Optimal for sampling from a shader
//
// *************************************************************************************************************************
//
//