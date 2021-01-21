#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragView;
layout(location = 3) in vec3 fragLight;
layout(location = 4) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = texture(texSampler, fragTexCoord);

	vec3 N = normalize(fragNormal);
	vec3 L = normalize(fragLight);
	vec3 V = normalize(fragView);
	vec3 R = reflect(L, N);

	vec4 ambient = texture(texSampler, fragTexCoord) * 0.1;
	vec4 diffuse = texture(texSampler, fragTexCoord) * max(dot(N, L), 0.0);
	vec3 specular = pow(max(dot(R, V), 0.0), 16.0) * vec3(1.35);

	outColor = vec4(ambient.xyz + diffuse.xyz + specular, 1.0);
}