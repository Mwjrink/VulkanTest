#pragma once

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>

#include <string>
#include <vector>

#include "Vertex.h"

class _Model
{
  private:
    std::string mesh_path    = "models/chalet.obj";
    std::string texture_path = "textures/chalet.jpg";

    uint32_t mipLevels;

	int      texWidth, texHeight;
    stbi_uc* pixels;

    VkDevice* device;

    // Move these to RenderGroup ...
    // VkBuffer       vertexBuffer;
    // VkDeviceMemory vertexBufferMemory;
    // VkBuffer       indexBuffer;
    // VkDeviceMemory indexBufferMemory;

    // TODO: @MaxCompleteAPI, this likely shouldn't be here and if it needs to be it should be dumped after loading
    // put this as a temp variable in loading then dump it once uploaded?
    // maybe not possible when we are recreating RenderGroups and need to reupload these to the buffer
    // std::vector<Vertex>   vertices;
    // std::vector<uint32_t> indices;  // It is possible to use either uint16_t or uint32_t for your index buffer depending
    // on
    // the number of entries in vertices. We can stick to uint16_t for now because we're
    // using less than 65535 unique vertices.
    size_t verticesOffset;
    size_t verticesCount;
    size_t indicesOffset;
    size_t indicesCount;

    std::vector<glm::mat4> _modelMatrices;

    bool cleaned_up = false;

    friend class VulkanApplication;
    friend class _RenderGroup;

  public:
    ~_Model()
    {
        // TODO: @MaxConcurrency, make this a compareExchange???
        // not sure if necessary though cause destructors might be thread-safe in the standard??
        if (!cleaned_up)
        {
            // vkDestroyImageView(*device, textureImageView, nullptr);

            // vkDestroyImage(*device, textureImage, nullptr);
            // vkFreeMemory(*device, textureImageMemory, nullptr);

            // vkDestroyBuffer(*device, indexBuffer, nullptr);
            // vkFreeMemory(*device, indexBufferMemory, nullptr);

            // vkDestroyBuffer(*device, vertexBuffer, nullptr);
            // vkFreeMemory(*device, vertexBufferMemory, nullptr);  // used by buffer, deleted after it

            cleaned_up = true;
        }
    }
};
