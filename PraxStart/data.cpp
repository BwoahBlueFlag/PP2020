// Project Practice 2020/2021 - Fast 3D scene rendering using LOD
// data.cpp - helper functions for vertex data type
// Author: Marek Janciar

#include "data.h"

#include <glm/gtx/hash.hpp>

vk::VertexInputBindingDescription Vertex::getBindingDescription()
{
    return vk::VertexInputBindingDescription{ 0, sizeof(Vertex), vk::VertexInputRate::eVertex };
}

std::array<vk::VertexInputAttributeDescription, 4> Vertex::getAttributeDescriptions()
{
    std::array<vk::VertexInputAttributeDescription, 4> attributeDescriptions{};

    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = vk::Format::eR32G32B32Sfloat;
    attributeDescriptions[0].offset = offsetof(Vertex, pos);

    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = vk::Format::eR32G32B32Sfloat;
    attributeDescriptions[1].offset = offsetof(Vertex, color);

    attributeDescriptions[2].binding = 0;
    attributeDescriptions[2].location = 2;
    attributeDescriptions[2].format = vk::Format::eR32G32B32Sfloat;
    attributeDescriptions[2].offset = offsetof(Vertex, normal);

    attributeDescriptions[3].binding = 0;
    attributeDescriptions[3].location = 3;
    attributeDescriptions[3].format = vk::Format::eR32G32Sfloat;
    attributeDescriptions[3].offset = offsetof(Vertex, texCoord);

    return attributeDescriptions;
}

bool Vertex::operator==(const Vertex& other) const
{
    return pos == other.pos && color == other.color && texCoord == other.texCoord && normal == other.normal;
}

bool Vertex::operator!=(const Vertex& other) const
{
    return pos != other.pos || color != other.color || texCoord != other.texCoord || normal != other.normal;
}

template<> struct std::hash<Vertex> {
    size_t operator()(Vertex const& vertex) const
    {
        std::size_t res = 0;
        hash_combine(res, vertex.pos);
        hash_combine(res, vertex.color);
        hash_combine(res, vertex.texCoord);
        hash_combine(res, vertex.normal);
        return res;
    }
};