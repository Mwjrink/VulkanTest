#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <map>
#ifdef _WIN32
#include <filesystem>
#include <optional>
#elif __APPLE__
#include <experimental/optional>
#endif
#include <algorithm>
#include <array>
#include <cstdint>  // Necessary for UINT32_MAX
#include <fstream>
#include <set>
#include <stdexcept>
#include <vector>

#include <iostream>

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

static std::vector<char> readFile(const std::string& filename)
{
    /*
    We start by opening the file with two flags:
    ate: // Start reading at the end of the file
    binary: // Read the file as binary file(avoid text transformations)
    */
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
    const int   WIDTH  = 800;
    const int   HEIGHT = 600;

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

    VkRenderPass     renderPass;
    VkPipelineLayout pipelineLayout;

    VkPipeline graphicsPipeline;

    std::vector<VkFramebuffer> swapChainFramebuffers;

    VkCommandPool commandPool;

    std::vector<VkCommandBuffer> commandBuffers;

    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence>     inFlightFences;
    size_t                   currentFrame = 0;

    const std::vector<const char*> validationLayers = {
        "VK_LAYER_LUNARG_standard_validation"};  // cant seem to find the macro for this

    const std::vector<const char*> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

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

#ifdef _WIN32
        // TODO: @MaxWindowsSpecific, find a way to make the directory if it does not alreay exist (probably add an ini
        // option for the path) also add a date-timestamp to this
        int logNumber = 0;
        while (std::filesystem::exists(logPath + "/Graphics/" + fileName))
        {
            fileName = "grapics" + std::to_string(logNumber) + ".log";
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
        createGraphicsPipeline();
        createFramebuffers();
        createCommandPool();
        createCommandBuffers();
        createSyncObjects();
    }

    void recreateSwapChain()
    {
        vkDeviceWaitIdle(device);

        cleanupSwapChain();

        createSwapChain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();
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

            // The actual vkCmdDraw function is a bit anticlimactic, but it's so simple because of all the information we
            // specified in advance. It has the following parameters, aside from the command buffer:
            // vertexCount: Even though we don't have a vertex buffer, we technically still have 3 vertices to draw.
            // instanceCount: Used for instanced rendering, use 1 if you're not doing that.
            // firstVertex: Used as an offset into the vertex buffer, defines the lowest value of gl_VertexIndex.
            // firstInstance: Used as an offset for instanced rendering, defines the lowest value of gl_InstanceIndex.
            vkCmdDraw(commandBuffers[i], 3, 1, 0, 0);

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

        /*
        Bindings: // spacing between data and whether the data is per-vertex or per-instance
                  // https://en.wikipedia.org/wiki/Geometry_instancing
        Attribute descriptions: // type of the attributes passed to the vertex shader,which binding to load them from and
        at
                                // which offset
        */
        VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
        vertexInputInfo.sType                                = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount        = 0;
        vertexInputInfo.pVertexBindingDescriptions           = nullptr;  // Optional
        vertexInputInfo.vertexAttributeDescriptionCount      = 0;
        vertexInputInfo.pVertexAttributeDescriptions         = nullptr;  // Optional

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
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;

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
        pipelineLayoutInfo.setLayoutCount             = 0;        // Optional
        pipelineLayoutInfo.pSetLayouts                = nullptr;  // Optional
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
        vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE,
                              &imageIndex);

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

        // The function takes an array of VkSubmitInfo structures as argument for efficiency when the workload is much
        // larger. The last parameter references an optional fence that will be signaled when the command buffers finish
        // execution.
        auto result = vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]);
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
        // throwOnError(result, "");

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
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

        for (size_t i = 0; i < swapChainImageViews.size(); i++)
        {
            vkDestroyImageView(device, swapChainImageViews[i], nullptr);
        }

        vkDestroySwapchainKHR(device, swapChain, nullptr);
    }

    void cleanup()
    {
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);
        }

        vkDestroyCommandPool(device, commandPool, nullptr);

        // Must be before the imageViews and renderPass
        for (auto framebuffer : swapChainFramebuffers)
        {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }

        vkDestroyPipeline(device, graphicsPipeline, nullptr);

        // Must be after graphicsPipeline
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);

        // Must be after pipeline layout
        vkDestroyRenderPass(device, renderPass, nullptr);

        for (auto imageView : swapChainImageViews)
        {
            vkDestroyImageView(device, imageView, nullptr);
        }

        // Must be before device
        vkDestroySwapchainKHR(device, swapChain, nullptr);

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
            initLog("");
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
//
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
//
//
// Also try running the compiler without any arguments to see what kinds of flags it supports. It can, for example, also
// output the bytecode into a human-readable format so you can see exactly what your shader is doing and any
// optimizations that have been applied at this stage.
//
//
//
// The Vulkan SDK includes libshaderc, which is a library to compile GLSL code to SPIR-V from within your program.
// https://github.com/google/shaderc
//
//
//
// TODO: @MaxUpgradeAPI, Look into custom allocators for all the objects we are creating through vulkan calls
//
//
//
// Fences are mainly designed to synchronize your application itself with rendering operation, whereas semaphores are
// used to synchronize operations within or across command queues.
//
//
//
// That's all it takes to recreate the swap chain! However, the disadvantage of this approach is that we need to stop all
// rendering before creating the new swap chain. It is possible to create a new swap chain while drawing commands on an image
// from the old swap chain are still in-flight. You need to pass the previous swap chain to the oldSwapChain field in the
// VkSwapchainCreateInfoKHR struct and destroy the old swap chain as soon as you've finished using it.
//
//
//
//