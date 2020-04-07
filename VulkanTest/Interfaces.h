#include <string>

#include "VulkanApplication.h"
#include "observer_ptr.h"

class Model;
class Instance;

class RenderGroup
{
  private:
    std::observer_ptr<VulkanApplication> _app;
    uint32_t                             _index;

  public:
    RenderGroup(const std::observer_ptr<VulkanApplication> app);

    Model addNewModel(const std::string &mesh_path, const std::string &texture_path);

    friend class Model;
    friend class Instance;
};

class Model
{
  private:
    std::observer_ptr<VulkanApplication> _app;
    uint32_t                             _index;
    uint32_t                             _rgIndex;

    Model(const std::observer_ptr<VulkanApplication> app, const uint32_t rgIndex, const std::string &mesh_path, const std::string &texture_path);

  public:
    Model(const std::observer_ptr<VulkanApplication> app, RenderGroup &renderGroup, const std::string &mesh_path, const std::string &texture_path);

    Instance addInstance();

    friend class RenderGroup;
    friend class Instance;
};

class Instance
{
  private:
    std::observer_ptr<VulkanApplication> _app;
    uint32_t                             _index;
    uint32_t                             _rgIndex;
    uint32_t                             _modelIndex;

    Instance(const std::observer_ptr<VulkanApplication> app, const uint32_t rgIndex, const uint32_t modelIndex);

  public:
    // const glm::mat4& modelMatrix = glm::mat4(1.0f)
    Instance(const std::observer_ptr<VulkanApplication> app, RenderGroup &renderGroup, Model &model);

    void updateModelMatrix(const glm::mat4 &model);

    glm::mat4* ModelMatrix();

    friend class Model;
    friend class RenderGroup;
};