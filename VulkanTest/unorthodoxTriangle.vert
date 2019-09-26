#version 450
#extension GL_ARB_separate_shader_objects : enable

// Vertex Data
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCoord;

// Instanced Data
layout(location = 3) in mat4 projViewModel;
layout(location = 4) in int textureIndex;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;
layout(location = 2) out int samplerIndex;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
    fragColor = inColor;
    fragTexCoord = inTexCoord;
    samplerIndex = textureIndex;
}
