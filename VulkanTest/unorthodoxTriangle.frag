#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 1) uniform sampler2DArray samplerArray; // THIS

layout(location = 0) in vec3 inColor;
layout(location = 1) in vec3 inUV;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = texture(samplerArray, inUV); // * inColor ???;
}
