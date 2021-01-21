// Project Practice 2020/2021 - Fast 3D scene rendering using LOD
// main.cpp - application based on vulkan for simplyfiyng and rendering 3D meshes
// Author: Marek Janciar

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>
#include <chrono>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <optional>
#include <set>
#include <algorithm>
#include <fstream>
#include <unordered_map>
#include "data.h"
#include "simply.h"

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;
const std::vector<const char*> validationLayers = { "VK_LAYER_KHRONOS_validation"};
const std::vector<const char*> deviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME};
const int MAX_FRAMES_IN_FLIGHT = 2;
const std::string MODEL_PATH = "models/body.obj";
const std::string TEXTURE_PATH = "textures/viking_room.png";

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

struct UniformBufferObject 
{
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
    glm::vec3 light;
};

class DemoApp
{
public:
    void run() 
    {
        lastFrame = std::chrono::high_resolution_clock::now();
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* window;
    
    vk::UniqueInstance instance;
    vk::PhysicalDevice physicalDevice;
    vk::UniqueDevice device;
    vk::Queue graphicsQueue;
    vk::UniqueSurfaceKHR surface;
    vk::Queue presentQueue;
    vk::UniqueCommandPool commandPool;
    std::vector<vk::UniqueSemaphore> imageAvailableSemaphores;
    std::vector<vk::UniqueSemaphore> renderFinishedSemaphores;
    std::vector<vk::UniqueFence> inFlightFences;
    std::vector<vk::Fence> imagesInFlight;
    vk::UniqueBuffer vertexBuffer;
    vk::UniqueDeviceMemory vertexBufferMemory;
    vk::UniqueBuffer indexBuffer;
    vk::UniqueDeviceMemory indexBufferMemory;
    vk::UniqueDescriptorSetLayout descriptorSetLayout;
    uint32_t mipLevels;
    vk::UniqueImage textureImage;
    vk::UniqueDeviceMemory textureImageMemory;
    vk::UniqueSampler textureSampler;
    vk::UniqueImageView textureImageView;
    
    std::vector<vk::Image> swapChainImages;
    vk::Format swapChainImageFormat;
    vk::Extent2D swapChainExtent;
    std::vector<vk::DescriptorSet> descriptorSets;
    
    vk::UniqueDescriptorPool descriptorPool;
    std::vector<vk::UniqueDeviceMemory> uniformBuffersMemory;
    std::vector<vk::UniqueBuffer> uniformBuffers;
    vk::UniqueSwapchainKHR swapChain;
    std::vector<vk::UniqueImageView> swapChainImageViews;
    vk::UniqueRenderPass renderPass;
    vk::UniquePipelineLayout pipelineLayout;
    vk::UniquePipeline graphicsPipeline;
    std::vector<vk::UniqueCommandBuffer> commandBuffers;
    std::vector<vk::UniqueFramebuffer> swapChainFramebuffers;
    vk::UniqueDeviceMemory depthImageMemory;
    vk::UniqueImage depthImage;
    vk::UniqueImageView depthImageView;
    vk::SampleCountFlagBits msaaSamples = vk::SampleCountFlagBits::e1;
    vk::UniqueImage colorImage;
    vk::UniqueDeviceMemory colorImageMemory;
    vk::UniqueImageView colorImageView;

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    size_t currentFrame = 0;
    bool framebufferResized = false;

    std::chrono::steady_clock::time_point lastFrame;

    void initVulkan()
    {
        createInstance();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createCommandPool();
        createColorResources();
        createDepthResources();
        createFramebuffers();
        createTextureImage();
        createTextureImageView();
        createTextureSampler();
        loadModel();

        std::cout << indices.size() / 3 << std::endl;
        std::cout << "Enter threshold value:" << std::endl;
        float user;
        std::cin >> user;
        LengthIncremental(vertices, user);
        //VertexClustering(vertices, user);
        indices.resize(vertices.size());
        std::cout << indices.size() / 3 << std::endl;

        createVertexBuffer();
        createIndexBuffer();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
        createSyncObjects();
    }

    void mainLoop()
    {
        while (!glfwWindowShouldClose(window))
        {
            glfwPollEvents();
            drawFrame();
        }

        device->waitIdle();
    }

    void loadModel()
    {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;

        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH.c_str()))
            throw std::runtime_error(warn + err);

        for (const auto& shape : shapes)
        {
            for (const auto& index : shape.mesh.indices)
            {
                Vertex vertex{};

                vertex.pos = {
                    attrib.vertices[3 * index.vertex_index + 0],
                    attrib.vertices[3 * index.vertex_index + 1],
                    attrib.vertices[3 * index.vertex_index + 2]
                };

                vertex.texCoord = { 1.0f, 1.0f
                    /*attrib.texcoords[2 * index.texcoord_index + 0],
                    1.0f - attrib.texcoords[2 * index.texcoord_index + 1]*/
                };

                vertex.color = { 0.5f, 0.5f, 0.5f };

                vertex.normal = {
                    attrib.normals[3 * index.normal_index + 0],
                    attrib.normals[3 * index.normal_index + 1],
                    attrib.normals[3 * index.normal_index + 2]
                };

                // Temporarily not deduplicating vertices for simplyfication functions
                /*if (uniqueVertices.count(vertex) == 0)
                {
                    uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
                    vertices.push_back(vertex);
                }

                indices.push_back(uniqueVertices[vertex]);*/

                vertices.push_back(vertex);
                indices.push_back(indices.size());
            }
        }
    }

    void initWindow() 
    {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    }

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) 
    {
        auto app = reinterpret_cast<DemoApp*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    void cleanup()
    {
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void createColorResources() 
    {
        vk::Format colorFormat = swapChainImageFormat;

        createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, colorFormat, vk::ImageTiling::eOptimal, 
            vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal, colorImage, colorImageMemory);
        colorImageView = createImageView(colorImage.get(), colorFormat, vk::ImageAspectFlagBits::eColor, 1);
    }

    vk::SampleCountFlagBits getMaxUsableSampleCount() 
    {
        vk::PhysicalDeviceProperties physicalDeviceProperties = physicalDevice.getProperties();

        vk::SampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts;
        if (counts & vk::SampleCountFlagBits::e64) 
            return vk::SampleCountFlagBits::e64;
        if (counts & vk::SampleCountFlagBits::e32) 
            return vk::SampleCountFlagBits::e32;
        if (counts & vk::SampleCountFlagBits::e16) 
            return vk::SampleCountFlagBits::e16;
        if (counts & vk::SampleCountFlagBits::e8) 
            return vk::SampleCountFlagBits::e8;
        if (counts & vk::SampleCountFlagBits::e4)
            return vk::SampleCountFlagBits::e4;
        if (counts & vk::SampleCountFlagBits::e2)
            return vk::SampleCountFlagBits::e2;

        return vk::SampleCountFlagBits::e1;
    }

    void generateMipmaps(vk::Image image, vk::Format imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels) 
    {
        vk::FormatProperties formatProperties = physicalDevice.getFormatProperties(imageFormat);

        if (!(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear))
            throw std::runtime_error("texture image format does not support linear blitting!");

        std::vector<vk::UniqueCommandBuffer> commandBuffer = beginSingleTimeCommands();

        vk::ImageMemoryBarrier barrier{};
        barrier.image = image;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.subresourceRange.levelCount = 1;

        int32_t mipWidth = texWidth;
        int32_t mipHeight = texHeight;

        for (uint32_t i = 1; i < mipLevels; i++)
        {
            barrier.subresourceRange.baseMipLevel = i - 1;
            barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
            barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;

            commandBuffer[0]->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, (vk::DependencyFlags) 0, 0, nullptr, 0, nullptr, 1, &barrier);

            vk::ImageBlit blit{};
            blit.srcOffsets[0] = { 0, 0, 0 };
            blit.srcOffsets[1] = { mipWidth, mipHeight, 1 };
            blit.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
            blit.srcSubresource.mipLevel = i - 1;
            blit.srcSubresource.baseArrayLayer = 0;
            blit.srcSubresource.layerCount = 1;
            blit.dstOffsets[0] = { 0, 0, 0 };
            blit.dstOffsets[1] = { mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1 };
            blit.dstSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
            blit.dstSubresource.mipLevel = i;
            blit.dstSubresource.baseArrayLayer = 0;
            blit.dstSubresource.layerCount = 1;

            commandBuffer[0]->blitImage(image, vk::ImageLayout::eTransferSrcOptimal, image, vk::ImageLayout::eTransferDstOptimal, 1, &blit, vk::Filter::eLinear);

            barrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
            barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

            commandBuffer[0]->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, (vk::DependencyFlags) 0, 0, nullptr, 0, nullptr, 1, &barrier);

            if (mipWidth > 1) mipWidth /= 2;
            if (mipHeight > 1) mipHeight /= 2;
        }

        barrier.subresourceRange.baseMipLevel = mipLevels - 1;
        barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
        barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        commandBuffer[0]->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, (vk::DependencyFlags) 0, 0, nullptr, 0, nullptr, 1, &barrier);

        endSingleTimeCommands(commandBuffer);
    }

    bool hasStencilComponent(vk::Format format) 
    {
        return format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint;
    }

    vk::Format findDepthFormat() 
    {
        return findSupportedFormat({ vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint }, vk::ImageTiling::eOptimal, 
            vk::FormatFeatureFlagBits::eDepthStencilAttachment);
    }

    vk::Format findSupportedFormat(const std::vector<vk::Format>& candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features) 
    {
        for (vk::Format format : candidates) 
        {
            vk::FormatProperties props = physicalDevice.getFormatProperties(format);

            if (tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features)
                return format;
            else if (tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features)
                return format;
        }

        throw std::runtime_error("failed to find supported format!");
    }

    void createDepthResources() 
    {
        vk::Format depthFormat = findDepthFormat();

        createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, depthFormat, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal,
            depthImage, depthImageMemory);
        depthImageView = createImageView(depthImage.get(), depthFormat, vk::ImageAspectFlagBits::eDepth, 1);
    }

    void createTextureSampler() 
    {
        vk::SamplerCreateInfo samplerInfo{vk::SamplerCreateFlags(), vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear, vk::SamplerAddressMode::eRepeat, 
            vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat, 0.0f, VK_TRUE, 16.0f, VK_FALSE, vk::CompareOp::eAlways, 0.0f, static_cast<float>(mipLevels), 
            vk::BorderColor::eIntOpaqueBlack, VK_FALSE};

        textureSampler = device->createSamplerUnique(samplerInfo);
    }

    vk::UniqueImageView createImageView(vk::Image image, vk::Format format, vk::ImageAspectFlags aspectFlags, uint32_t mipLevels)
    {
        vk::ImageViewCreateInfo createInfo{ vk::ImageViewCreateFlags(), image, vk::ImageViewType::e2D, format, vk::ComponentMapping{
                vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity}, vk::ImageSubresourceRange{
                aspectFlags, 0, mipLevels, 0, 1} };

        return device->createImageViewUnique(createInfo, nullptr);
    }

    void createTextureImageView() 
    {
        textureImageView = createImageView(textureImage.get(), vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor, mipLevels);
    }

    void copyBufferToImage(vk::UniqueBuffer& buffer, vk::UniqueImage& image, uint32_t width, uint32_t height)
    {
        std::vector<vk::UniqueCommandBuffer> commandBuffer = beginSingleTimeCommands();

        vk::BufferImageCopy region{0, 0, 0, vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1}, { 0, 0, 0 }, { width, height, 1} };

        commandBuffer[0]->copyBufferToImage(buffer.get(), image.get(), vk::ImageLayout::eTransferDstOptimal, 1, &region);

        endSingleTimeCommands(commandBuffer);
    }

    void transitionImageLayout(vk::UniqueImage& image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout, uint32_t mipLevels)
    {
        std::vector<vk::UniqueCommandBuffer> commandBuffer = beginSingleTimeCommands();

        vk::ImageMemoryBarrier barrier{ (vk::AccessFlags) 0, (vk::AccessFlags) 0, oldLayout, newLayout, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image.get(),
            vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, mipLevels, 0, 1} };
        
        vk::PipelineStageFlags sourceStage;
        vk::PipelineStageFlags destinationStage;

        if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) 
        {
            barrier.srcAccessMask = (vk::AccessFlags) 0;
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

            sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
            destinationStage = vk::PipelineStageFlagBits::eTransfer;
        }
        else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
        {
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

            sourceStage = vk::PipelineStageFlagBits::eTransfer;
            destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
        }
        else 
            throw std::invalid_argument("unsupported layout transition!");

        commandBuffer[0]->pipelineBarrier(sourceStage, destinationStage, (vk::DependencyFlags) 0, 0, nullptr, 0, nullptr, 1, &barrier);

        endSingleTimeCommands(commandBuffer);
    }

    std::vector<vk::UniqueCommandBuffer> beginSingleTimeCommands()
    {
        std::vector<vk::UniqueCommandBuffer> commandBuffer = device->allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo{ commandPool.get(), vk::CommandBufferLevel::ePrimary, 1 });

        commandBuffer[0]->begin(vk::CommandBufferBeginInfo{ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });

        return commandBuffer;
    }

    void endSingleTimeCommands(std::vector<vk::UniqueCommandBuffer>& commandBuffer)
    {
        commandBuffer[0]->end();

        vk::SubmitInfo submitInfo{};
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer[0].get();

        graphicsQueue.submit(1, &submitInfo, nullptr);
        graphicsQueue.waitIdle();
    }

    void createImage(uint32_t width, uint32_t height, uint32_t mipLevels, vk::SampleCountFlagBits numSamples, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, 
        vk::MemoryPropertyFlags properties, vk::UniqueImage& image, vk::UniqueDeviceMemory& imageMemory) 
    {
        vk::ImageCreateInfo imageInfo{ vk::ImageCreateFlags(), vk::ImageType::e2D, format, vk::Extent3D{width, height, 1}, mipLevels, 1, numSamples, tiling, usage, 
            vk::SharingMode::eExclusive };
        imageInfo.initialLayout = vk::ImageLayout::eUndefined;

        image = device->createImageUnique(imageInfo);

        vk::MemoryRequirements memRequirements = device->getImageMemoryRequirements(image.get());

        vk::MemoryAllocateInfo allocInfo{ memRequirements.size, findMemoryType(memRequirements.memoryTypeBits, properties) };

        imageMemory = device->allocateMemoryUnique(allocInfo);

        device->bindImageMemory(image.get(), imageMemory.get(), 0);
    }

    void createTextureImage() 
    {
        int texWidth, texHeight, texChannels;
        stbi_uc* pixels = stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        vk::DeviceSize imageSize = texWidth * texHeight * 4;
        mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;

        if (!pixels)
            throw std::runtime_error("failed to load texture image!");

        vk::UniqueBuffer stagingBuffer;
        vk::UniqueDeviceMemory stagingBufferMemory;

        createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

        void* data = device->mapMemory(stagingBufferMemory.get(), 0, imageSize);
        memcpy(data, pixels, static_cast<size_t>(imageSize));
        device->unmapMemory(stagingBufferMemory.get());

        stbi_image_free(pixels);

        createImage(static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight), mipLevels, vk::SampleCountFlagBits::e1, vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal, textureImage, textureImageMemory);

        transitionImageLayout(textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, mipLevels);
        copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
        generateMipmaps(textureImage.get(), vk::Format::eR8G8B8A8Srgb, texWidth, texHeight, mipLevels);
    }

    void createDescriptorSets() 
    {
        std::vector<vk::DescriptorSetLayout> layouts(swapChainImages.size(), descriptorSetLayout.get());
        vk::DescriptorSetAllocateInfo allocInfo{descriptorPool.get(), static_cast<uint32_t>(swapChainImages.size()), layouts.data()};

        descriptorSets = device->allocateDescriptorSets(allocInfo);

        for (size_t i = 0; i < swapChainImages.size(); i++) 
        {
            vk::DescriptorBufferInfo bufferInfo{ uniformBuffers[i].get(), 0, sizeof(UniformBufferObject) };

            vk::DescriptorImageInfo imageInfo{textureSampler.get(), textureImageView.get(), vk::ImageLayout::eShaderReadOnlyOptimal};

            std::array<vk::WriteDescriptorSet, 2> descriptorWrites{ vk::WriteDescriptorSet{ descriptorSets[i], 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &bufferInfo, nullptr}, 
                vk::WriteDescriptorSet{ descriptorSets[i], 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &imageInfo, nullptr, nullptr} };

            device->updateDescriptorSets(static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
        }
    }

    void createDescriptorPool() 
    {
        std::array<vk::DescriptorPoolSize, 2> poolSizes{ vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, static_cast<uint32_t>(swapChainImages.size())}, 
            vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, static_cast<uint32_t>(swapChainImages.size())} };

        vk::DescriptorPoolCreateInfo poolInfo{vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, static_cast<uint32_t>(swapChainImages.size()), static_cast<uint32_t>(poolSizes.size()), poolSizes.data()};
        
        descriptorPool = device->createDescriptorPoolUnique(poolInfo);
    }

    void createDescriptorSetLayout() 
    {
        vk::DescriptorSetLayoutBinding samplerLayoutBinding{1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment, nullptr};

        vk::DescriptorSetLayoutBinding uboLayoutBinding{0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex, nullptr};
        
        std::array<vk::DescriptorSetLayoutBinding, 2> bindings = { uboLayoutBinding, samplerLayoutBinding };

        vk::DescriptorSetLayoutCreateInfo layoutInfo{vk::DescriptorSetLayoutCreateFlags(), static_cast<uint32_t>(bindings.size()), bindings.data() };
        
        descriptorSetLayout = device->createDescriptorSetLayoutUnique(layoutInfo);
    }

    void createUniformBuffers() 
    {
        vk::DeviceSize bufferSize = sizeof(UniformBufferObject);

        uniformBuffers.resize(swapChainImages.size());
        uniformBuffersMemory.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++)
            createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, uniformBuffers[i], 
                uniformBuffersMemory[i]);
    }

    void createIndexBuffer() 
    {
        vk::UniqueBuffer stagingBuffer;
        vk::UniqueDeviceMemory stagingBufferMemory;

        createBuffer(sizeof(indices[0]) * indices.size(), vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer,
            stagingBufferMemory);

        void* data;
        data = device->mapMemory(stagingBufferMemory.get(), 0, sizeof(indices[0]) * indices.size());
        memcpy(data, indices.data(), sizeof(indices[0]) * indices.size());
        device->unmapMemory(stagingBufferMemory.get());

        createBuffer(sizeof(indices[0]) * indices.size(), vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, indexBuffer,
            indexBufferMemory);

        copyBuffer(stagingBuffer.get(), indexBuffer.get(), sizeof(indices[0]) * indices.size());
    }

    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) 
    {
        vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) 
        {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
                return i;
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::UniqueBuffer& buffer, vk::UniqueDeviceMemory& bufferMemory)
    {
        buffer = device->createBufferUnique(vk::BufferCreateInfo{ vk::BufferCreateFlags(), size, usage, vk::SharingMode::eExclusive });

        vk::MemoryRequirements memRequirements = device->getBufferMemoryRequirements(buffer.get());

        vk::MemoryAllocateInfo allocInfo{ memRequirements.size, findMemoryType(memRequirements.memoryTypeBits, properties) };

        bufferMemory = device->allocateMemoryUnique(vk::MemoryAllocateInfo{ memRequirements.size, findMemoryType(memRequirements.memoryTypeBits, properties) });

        device->bindBufferMemory(buffer.get(), bufferMemory.get(), 0);
    }

    void copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size) 
    {
        std::vector<vk::UniqueCommandBuffer> commandBuffer = beginSingleTimeCommands();

        vk::BufferCopy copyRegion{0, 0, size};
        commandBuffer[0]->copyBuffer(srcBuffer, dstBuffer, 1, &copyRegion);

        endSingleTimeCommands(commandBuffer);
    }

    void createVertexBuffer() 
    {
        vk::UniqueBuffer stagingBuffer;
        vk::UniqueDeviceMemory stagingBufferMemory;
        
        createBuffer(sizeof(vertices[0]) * vertices.size(), vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, 
            stagingBufferMemory);

        void* data;
        data = device->mapMemory(stagingBufferMemory.get(), 0, sizeof(vertices[0]) * vertices.size());
        memcpy(data, vertices.data(), sizeof(vertices[0]) * vertices.size());
        device->unmapMemory(stagingBufferMemory.get());

        createBuffer(sizeof(vertices[0]) * vertices.size(), vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, vertexBuffer, 
            vertexBufferMemory);

        copyBuffer(stagingBuffer.get(), vertexBuffer.get(), sizeof(vertices[0]) * vertices.size());
    }

    void createSyncObjects()
    {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
        imagesInFlight.resize(swapChainImages.size(), nullptr);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) 
        {
            imageAvailableSemaphores[i] = device->createSemaphoreUnique(vk::SemaphoreCreateInfo{});
            renderFinishedSemaphores[i] = device->createSemaphoreUnique(vk::SemaphoreCreateInfo{});
            inFlightFences[i] = device->createFenceUnique(vk::FenceCreateInfo{vk::FenceCreateFlagBits::eSignaled});
        }
    }

    void createCommandBuffers() 
    {
        commandBuffers.resize(swapChainFramebuffers.size());

        commandBuffers = device->allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo{ commandPool.get(), vk::CommandBufferLevel::ePrimary, (uint32_t)commandBuffers.size() });

        for (size_t i = 0; i < commandBuffers.size(); i++) 
        {
            commandBuffers[i]->begin(vk::CommandBufferBeginInfo{ {}, nullptr });

            std::array<vk::ClearValue, 2> clearValues{};
            clearValues[0].color = vk::ClearColorValue{ std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 1.0f } };
            clearValues[1].depthStencil = { 1.0f, 0 };
            
            commandBuffers[i]->beginRenderPass(vk::RenderPassBeginInfo{ renderPass.get(), swapChainFramebuffers[i].get(), vk::Rect2D{ {0,0}, swapChainExtent }, 
                static_cast<uint32_t>(clearValues.size()), clearValues.data() }, vk::SubpassContents::eInline);
            commandBuffers[i]->bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline.get());

            vk::Buffer vertexBuffers[] = { vertexBuffer.get() };
            vk::DeviceSize offsets[] = { 0 };
            commandBuffers[i]->bindVertexBuffers(0, 1, vertexBuffers, offsets);

            commandBuffers[i]->bindIndexBuffer(indexBuffer.get(), 0, vk::IndexType::eUint32);

            commandBuffers[i]->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout.get(), 0, 1, &descriptorSets[i], 0, nullptr);

            commandBuffers[i]->drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
            commandBuffers[i]->endRenderPass();
            commandBuffers[i]->end();
        } 
    }

    void createCommandPool() 
    {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        vk::CommandPoolCreateInfo poolInfo{vk::CommandPoolCreateFlags(), queueFamilyIndices.graphicsFamily.value() };

        commandPool = device->createCommandPoolUnique(vk::CommandPoolCreateInfo{ vk::CommandPoolCreateFlags(), queueFamilyIndices.graphicsFamily.value() }, nullptr);
    }

    void createFramebuffers() 
    {
        swapChainFramebuffers.resize(swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); i++) 
        {
            std::array<vk::ImageView, 3> attachments = { colorImageView.get(), depthImageView.get(), swapChainImageViews[i].get()};

            vk::FramebufferCreateInfo framebufferInfo{vk::FramebufferCreateFlags(), renderPass.get(), static_cast<uint32_t>(attachments.size()), attachments.data(), swapChainExtent.width, 
                swapChainExtent.height, 1};

            swapChainFramebuffers[i] = device->createFramebufferUnique(
                vk::FramebufferCreateInfo { vk::FramebufferCreateFlags(), renderPass.get(), static_cast<uint32_t>(attachments.size()), attachments.data(), swapChainExtent.width, swapChainExtent.height, 1 }, nullptr);
        }
    }

    void createRenderPass() 
    {
        vk::SubpassDependency dependency{ VK_SUBPASS_EXTERNAL, 0, vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eColorAttachmentOutput, (vk::AccessFlagBits) 0, 
            vk::AccessFlagBits::eColorAttachmentWrite};
        
        vk::AttachmentDescription colorAttachment{ {}, swapChainImageFormat, msaaSamples, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare,
            vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal };

        vk::AttachmentReference colorAttachmentRef{0, vk::ImageLayout::eColorAttachmentOptimal};

        vk::AttachmentDescription colorAttachmentResolve{ {}, swapChainImageFormat, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eStore, 
            vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::ePresentSrcKHR };

        vk::AttachmentReference colorAttachmentResolveRef{2, vk::ImageLayout::eColorAttachmentOptimal};

        vk::AttachmentDescription depthAttachment{ {}, findDepthFormat(), msaaSamples, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eDontCare, vk::AttachmentLoadOp::eDontCare, 
            vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal};

        vk::AttachmentReference depthAttachmentRef{1, vk::ImageLayout::eDepthStencilAttachmentOptimal};

        vk::SubpassDescription subpass{ {}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &colorAttachmentRef, &colorAttachmentResolveRef, &depthAttachmentRef, 0, nullptr};

        std::array<vk::AttachmentDescription, 3> attachments = { colorAttachment, depthAttachment, colorAttachmentResolve };

        vk::RenderPassCreateInfo renderPassInfo{vk::RenderPassCreateFlags(), static_cast<uint32_t>(attachments.size()), attachments.data(), 1, &subpass, 1, &dependency};

        renderPass = device->createRenderPassUnique(renderPassInfo);
    }

    vk::UniqueShaderModule createShaderModule(const std::vector<char>& code) 
    {
        vk::ShaderModuleCreateInfo createInfo{vk::ShaderModuleCreateFlags(), code.size(), reinterpret_cast<const uint32_t*>(code.data()) };
        
        return device->createShaderModuleUnique(createInfo, nullptr);
    }

    void createGraphicsPipeline() 
    {
        auto vertShaderCode = readFile("shaders/vert.spv");
        auto fragShaderCode = readFile("shaders/frag.spv");

        vk::UniqueShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        vk::UniqueShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        vk::PipelineShaderStageCreateInfo vertShaderStageInfo{vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eVertex, vertShaderModule.get(), "main", nullptr};
        vk::PipelineShaderStageCreateInfo fragShaderStageInfo{vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eFragment, fragShaderModule.get(), "main", nullptr};
        vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{vk::PipelineVertexInputStateCreateFlags(), 1, &bindingDescription, static_cast<uint32_t>(attributeDescriptions.size()), 
            attributeDescriptions.data() };

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{vk::PipelineInputAssemblyStateCreateFlags(), vk::PrimitiveTopology::eTriangleList, VK_FALSE};

        vk::Viewport viewport{0.0f, 0.0f, (float)swapChainExtent.width, (float)swapChainExtent.height, 0.0f, 1.0f };

        vk::Rect2D scissor{ { 0, 0 } , swapChainExtent};

        vk::PipelineViewportStateCreateInfo viewportState{vk::PipelineViewportStateCreateFlags(), 1, &viewport, 1, &scissor};

        vk::PipelineRasterizationStateCreateInfo rasterizer{vk::PipelineRasterizationStateCreateFlags(), VK_FALSE, VK_FALSE, vk::PolygonMode::eFill, vk::CullModeFlagBits::eBack, 
            vk::FrontFace::eCounterClockwise, VK_FALSE, 0.0f, 0.0f, 0.0f, 1.0f};

        vk::PipelineMultisampleStateCreateInfo multisampling{vk::PipelineMultisampleStateCreateFlags(), msaaSamples, VK_FALSE, 1.0f, nullptr, VK_FALSE, VK_FALSE};

        vk::PipelineColorBlendAttachmentState colorBlendAttachment{VK_FALSE, vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd, vk::BlendFactor::eOne, vk::BlendFactor::eZero, 
            vk::BlendOp::eAdd, vk::ColorComponentFlagBits::eA | vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB };

        vk::PipelineColorBlendStateCreateInfo colorBlending{ vk::PipelineColorBlendStateCreateFlags(), VK_FALSE, vk::LogicOp::eCopy, 1, &colorBlendAttachment, {0.0f, 0.0f, 0.0f, 0.0f} };

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{vk::PipelineLayoutCreateFlags(), 1, &descriptorSetLayout.get(), 0, nullptr};

        pipelineLayout = device->createPipelineLayoutUnique(pipelineLayoutInfo, nullptr);

        vk::PipelineDepthStencilStateCreateInfo depthStencil{ vk::PipelineDepthStencilStateCreateFlags(), VK_TRUE, VK_TRUE, vk::CompareOp::eLess, VK_FALSE, VK_FALSE, vk::StencilOpState{},
            vk::StencilOpState{}, 0.0f, 1.0f };

        vk::GraphicsPipelineCreateInfo pipelineInfo{ {}, 2, shaderStages, &vertexInputInfo, &inputAssembly, nullptr, &viewportState, &rasterizer, &multisampling, &depthStencil, &colorBlending, 
            nullptr, pipelineLayout.get(), renderPass.get(), 0, nullptr, -1};

        graphicsPipeline = device->createGraphicsPipelineUnique(nullptr, pipelineInfo, nullptr);
    }

    void createImageViews() 
    {
        swapChainImageViews.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++) 
            swapChainImageViews[i] = createImageView(swapChainImages[i], swapChainImageFormat, vk::ImageAspectFlagBits::eColor, 1);
    }

    struct SwapChainSupportDetails
    {
        vk::SurfaceCapabilitiesKHR capabilities;
        std::vector<vk::SurfaceFormatKHR> formats;
        std::vector<vk::PresentModeKHR> presentModes;
    };

    SwapChainSupportDetails querySwapChainSupport(vk::PhysicalDevice device)
    {
        SwapChainSupportDetails details;

        details.capabilities = device.getSurfaceCapabilitiesKHR(surface.get());
        details.formats = device.getSurfaceFormatsKHR(surface.get());
        details.presentModes = device.getSurfacePresentModesKHR(surface.get());

        return details;
    }

    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats)
    {
        for (const auto& availableFormat : availableFormats)
        {
            if (availableFormat.format == vk::Format::eB8G8R8A8Srgb && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
            {
                return availableFormat;
            }
        }

        return availableFormats[0];
    }

    vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes)
    {
        for (const auto& availablePresentMode : availablePresentModes)
        {
            if (availablePresentMode == vk::PresentModeKHR::eMailbox)
            {
                return availablePresentMode;
            }
        }

        return vk::PresentModeKHR::eFifo;
    }

    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) 
    {
        if (capabilities.currentExtent.width != UINT32_MAX) 
        {
            return capabilities.currentExtent;
        }
        else 
        {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            VkExtent2D actualExtent = { width, height };

            actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
            actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));

            return actualExtent;
        }
    }

    struct QueueFamilyIndices
    {
        std::optional<uint32_t> graphicsFamily;
        std::optional<uint32_t> presentFamily;

        bool isComplete()
        {
            return graphicsFamily.has_value() && presentFamily.has_value();
        }
    };

    QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice device)
    {
        QueueFamilyIndices indices;

        std::vector<vk::QueueFamilyProperties> queueFamilies = device.getQueueFamilyProperties();

        int i = 0;
        for (const auto& queueFamily : queueFamilies)
        {
            vk::Bool32 presentSupport = false;
            
            presentSupport = device.getSurfaceSupportKHR(i, surface.get());

            if (presentSupport) {
                indices.presentFamily = i;
            }

            if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics)
            {
                indices.graphicsFamily = i;
            }

            if (indices.isComplete())
            {
                break;
            }

            i++;
        }

        return indices;
    }

    void createSwapChain() 
    {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

        vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        vk::PresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        vk::Extent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) 
        {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        vk::SwapchainCreateInfoKHR createInfo{vk::SwapchainCreateFlagsKHR(), surface.get(), imageCount, surfaceFormat.format, surfaceFormat.colorSpace, extent, 1, 
            vk::ImageUsageFlagBits::eColorAttachment};

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        if (indices.graphicsFamily != indices.presentFamily) 
        {
            createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        else 
        {
            createInfo.imageSharingMode = vk::SharingMode::eExclusive;
            createInfo.queueFamilyIndexCount = 0; // Optional
            createInfo.pQueueFamilyIndices = nullptr; // Optional
        }

        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;
        createInfo.oldSwapchain = nullptr;

        swapChain = device->createSwapchainKHRUnique(createInfo, nullptr);

        swapChainImages = device->getSwapchainImagesKHR(swapChain.get());
        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }

    void createSurface() 
    {
        VkSurfaceKHR tmpSurface;
        
        if (glfwCreateWindowSurface(instance.get(), window, nullptr, &tmpSurface) != VK_SUCCESS) 
        {
            throw std::runtime_error("failed to create window surface!");
        }

        vk::ObjectDestroy<vk::Instance, vk::DispatchLoaderStatic> surfaceDeleter(instance.get());
        surface = vk::UniqueSurfaceKHR(tmpSurface, surfaceDeleter);
    }

    void createLogicalDevice() 
    {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies) 
        {
            vk::DeviceQueueCreateInfo queueCreateInfo{vk::DeviceQueueCreateFlags(), queueFamily, 1, &queuePriority};
            queueCreateInfos.push_back(queueCreateInfo);
        }

        vk::PhysicalDeviceFeatures deviceFeatures{};
        deviceFeatures.samplerAnisotropy = VK_TRUE;

        vk::DeviceCreateInfo createInfo{vk::DeviceCreateFlags(), static_cast<uint32_t>(queueCreateInfos.size()), queueCreateInfos.data()};

        createInfo.pEnabledFeatures = &deviceFeatures;

        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        if (enableValidationLayers) 
        {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else 
        {
            createInfo.enabledLayerCount = 0;
        }

        device = physicalDevice.createDeviceUnique(createInfo, nullptr);

        graphicsQueue = device->getQueue(indices.graphicsFamily.value(), 0);
        presentQueue = device->getQueue(indices.presentFamily.value(), 0);
    }

    bool checkDeviceExtensionSupport(vk::PhysicalDevice device) 
    {
        std::vector<vk::ExtensionProperties> availableExtensions = device.enumerateDeviceExtensionProperties(nullptr);

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for (const auto& extension : availableExtensions) 
        {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }

    bool isDeviceSuitable(vk::PhysicalDevice device)
    {
        QueueFamilyIndices indices = findQueueFamilies(device);

        bool extensionsSupported = checkDeviceExtensionSupport(device);

        bool swapChainAdequate = false;
        if (extensionsSupported) 
        {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

        return indices.isComplete() && extensionsSupported && swapChainAdequate && device.getFeatures().samplerAnisotropy;
    }

    void pickPhysicalDevice() 
    {
        std::vector<vk::PhysicalDevice> devices = instance->enumeratePhysicalDevices();

        for (const auto& device : devices) 
        {
            if (isDeviceSuitable(device)) 
            {
                physicalDevice = device;
                msaaSamples = getMaxUsableSampleCount();
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE) 
        {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }

    void drawFrame() 
    {
        std::chrono::steady_clock::time_point now = std::chrono::high_resolution_clock::now();

        //std::cout << 1 / std::chrono::duration<float, std::chrono::seconds::period>(now - lastFrame).count() << std::endl;

        lastFrame = now;
        
        device->waitForFences(1, &(inFlightFences[currentFrame].get()), VK_TRUE, UINT64_MAX);
        
        uint32_t imageIndex;

        try 
        {
            imageIndex = device->acquireNextImageKHR(swapChain.get(), UINT64_MAX, imageAvailableSemaphores[currentFrame].get(), nullptr).value;
        }
        catch (const vk::OutOfDateKHRError& e)
        {
            recreateSwapChain();
            return;
        }

        // Check if a previous frame is using this image (i.e. there is its fence to wait on)
        if (imagesInFlight[imageIndex] != VK_NULL_HANDLE)
            device->waitForFences(1, &(imagesInFlight[currentFrame]), VK_TRUE, UINT64_MAX);
        
        // Mark the image as now being in use by this frame
        imagesInFlight[imageIndex] = inFlightFences[currentFrame].get();

        vk::Semaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame].get()};
        vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput };
        vk::Semaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame].get() };

        updateUniformBuffer(imageIndex);

        vk::SubmitInfo submitInfo{1, waitSemaphores, waitStages, 1, &(commandBuffers[imageIndex].get()), 1, signalSemaphores};
        
        device->resetFences(1, &(inFlightFences[currentFrame].get()));

        graphicsQueue.submit(1, &submitInfo, inFlightFences[currentFrame].get());

        vk::SwapchainKHR swapChains[] = { swapChain.get() };

        vk::PresentInfoKHR presentInfo{1, signalSemaphores, 1, swapChains, &imageIndex, nullptr};

        try
        {
            if (presentQueue.presentKHR(&presentInfo) == vk::Result::eSuboptimalKHR || framebufferResized)
            {
                framebufferResized = false;
                recreateSwapChain();
            }
        }
        catch (const vk::OutOfDateKHRError& e)
        {
            recreateSwapChain();
        }

        presentQueue.waitIdle();

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void updateUniformBuffer(uint32_t currentImage) 
    {
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

        UniformBufferObject ubo{};

        ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f) * 0.5f, glm::vec3(0.0f, 0.0f, 1.0f));

        ubo.view = glm::lookAt(glm::vec3(0.0f, 4.0f, 3.0f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 0.0f, 1.0f));

        ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 10.0f);

        ubo.proj[1][1] *= -1;

        ubo.light = glm::vec3(10.0f, 10.0f, 10.0f);

        void* data = device->mapMemory(uniformBuffersMemory[currentImage].get(), 0, sizeof(ubo));
        memcpy(data, &ubo, sizeof(ubo));
        device->unmapMemory(uniformBuffersMemory[currentImage].get());
    }

    void createInstance() 
    {
        if (enableValidationLayers && !checkValidationLayerSupport()) 
        {
            throw std::runtime_error("validation layers requested, but not available!");
        }
        
        vk::ApplicationInfo appInfo{ "Hello Triangle", VK_MAKE_VERSION(1, 0, 0), "No Engine", VK_MAKE_VERSION(1, 0, 0), VK_API_VERSION_1_0 };

        vk::InstanceCreateInfo createInfo{};
        
        if (enableValidationLayers) 
        {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else 
        {
            createInfo.enabledLayerCount = 0;
        }

        createInfo.pApplicationInfo = &appInfo;

        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;

        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        createInfo.enabledExtensionCount = glfwExtensionCount;
        createInfo.ppEnabledExtensionNames = glfwExtensions;

        instance = vk::createInstanceUnique(createInfo, nullptr);

        std::vector<vk::ExtensionProperties> extensions = vk::enumerateInstanceExtensionProperties(nullptr);

        std::cout << "available extensions:\n";

        for (const auto& extension : extensions) 
        {
            std::cout << '\t' << extension.extensionName << '\n';
        }
    }

    bool checkValidationLayerSupport() 
    {
        std::vector<vk::LayerProperties> availableLayers = vk::enumerateInstanceLayerProperties();

        for (const char* layerName : validationLayers) 
        {
            bool layerFound = false;

            for (const auto& layerProperties : availableLayers) 
            {
                if (strcmp(layerName, layerProperties.layerName) == 0) 
                {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound) 
            {
                return false;
            }
        }

        return true;
    }

    void recreateSwapChain() 
    {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) 
        {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }
        
        device->waitIdle();

        swapChain.reset();

        createSwapChain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createColorResources();
        createDepthResources();
        createFramebuffers();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
    }

    static std::vector<char> readFile(const std::string& filename)
    {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open())
            throw std::runtime_error("failed to open file!");

        size_t fileSize = (size_t)file.tellg();
        std::vector<char> buffer(fileSize);
        file.seekg(0);
        file.read(buffer.data(), fileSize);
        file.close();

        return buffer;
    }
};

int main() 
{
    DemoApp app;

    try 
    {
        app.run();
    } 
    catch (vk::Error& e) 
    {
        std::cout << "Failed because of Vulkan exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch (std::exception& e) 
    {
        std::cout << "Failed because of exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch (...) 
    {
        std::cout << "Failed because of unspecified exception." << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}