#version 450
#extension GL_ARB_separate_shader_objects : enable

// Vertex Data
layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inUV;

// Instanced Data
layout(location = 3) in mat4 projViewModel;
layout(location = 4) in int arrayIndex;

layout(location = 0) out vec3 outColor;
layout(location = 1) out vec3 outUV;

void main() {
    gl_Position = projViewModel * vec4(inPos, 1.0);
    outColor = inColor;
    outUV = vec3(inUV, arrayIndex);
}
