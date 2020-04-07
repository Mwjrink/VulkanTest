#include "Interfaces.h"

Instance::Instance(const std::observer_ptr<VulkanApplication> app, const uint32_t rgIndex, const uint32_t modelIndex)
{
    _app        = app;
    _rgIndex    = rgIndex;
    _modelIndex = modelIndex;
    _index      = app->addInstance(_rgIndex, _modelIndex, glm::mat4(1.0f));
}

// const glm::mat4& modelMatrix = glm::mat4(1.0f)
Instance::Instance(const std::observer_ptr<VulkanApplication> app, RenderGroup &renderGroup, Model &model)
{
    _app        = app;
    _rgIndex    = renderGroup._index;
    _modelIndex = model._index;
    _index      = app->addInstance(_rgIndex, _modelIndex, glm::mat4(1.0f));
}

void Instance::updateModelMatrix(const glm::mat4 &model)
{
    _app->updateInstanceModelMatrix(_rgIndex, _modelIndex, _index, model);
}

glm::mat4* Instance::ModelMatrix() { _app->getModelMatrix(_rgIndex, _modelIndex, _index); }

Model::Model(const std::observer_ptr<VulkanApplication> app, const uint32_t rgIndex, std::string &mesh_path, std::string &texture_path)
{
    _app     = app;
    _rgIndex = rgIndex;
    _index   = app->addModel(_rgIndex, mesh_path, texture_path);
}

Model::Model(const std::observer_ptr<VulkanApplication> app, RenderGroup &renderGroup, std::string &mesh_path,
                std::string &texture_path)
{
    _app     = app;
    _rgIndex = renderGroup._index;
    _index   = app->addModel(_rgIndex, mesh_path, texture_path);
}

Instance Model::addInstance() { return Instance(_app, _rgIndex, _index); }

RenderGroup::RenderGroup(const std::observer_ptr<VulkanApplication> app)
{
    _app   = app;
    _index = app->createRenderGroup();
}

Model RenderGroup::addNewModel(const std::string &mesh_path, const std::string &texture_path)
{
    return Model(_app, _index, mesh_path, texture_path);
}