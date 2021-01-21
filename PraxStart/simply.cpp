// Project Practice 2020/2021 - Fast 3D scene rendering using LOD
// simply.cpp - implementation of mesh simplification methods 
// Author: Marek Janciar

#include "simply.h"
#include <unordered_map>
#include <map>
#include <glm/gtx/hash.hpp>
#include <iostream>
#include <unordered_set>

//---------------------------------------------------Edge-collapse---------------------------------------------------------------------------------------------
struct Edge
{
    float length;
    std::pair<size_t, size_t> vertices;
};

float DistanceSquared(glm::vec3 a, glm::vec3 b)
{
    a -= b;
    return dot(a, a);
}

void RemoveTriangle(size_t index, std::map<float, std::unordered_set<Edge*>>& orderedEdges, std::unordered_map<glm::vec3, std::unordered_set<size_t>>& samePosition,
    std::vector<Vertex>& inputVertices, std::vector<Edge*>& edgesA)
{
    samePosition[inputVertices[index].pos].erase(index);
    samePosition[inputVertices[index + 1].pos].erase(index + 1);
    samePosition[inputVertices[index + 2].pos].erase(index + 2);

    orderedEdges[edgesA[index]->length].erase(edgesA[index]);
    if (orderedEdges[edgesA[index]->length].empty())
        orderedEdges.erase(edgesA[index]->length);
    
    orderedEdges[edgesA[index + 1]->length].erase(edgesA[index + 1]);
    if (orderedEdges[edgesA[index + 1]->length].empty())
        orderedEdges.erase(edgesA[index + 1]->length);
    
    orderedEdges[edgesA[index + 2]->length].erase(edgesA[index + 2]);
    if (orderedEdges[edgesA[index + 2]->length].empty())
        orderedEdges.erase(edgesA[index + 2]->length);

    delete edgesA[index];
    delete edgesA[index + 1];
    delete edgesA[index + 2];

    edgesA[index] = nullptr;
    edgesA[index + 1] = nullptr;
    edgesA[index + 2] = nullptr;
}

void LengthIncremental(std::vector<Vertex>& inputVertices, float length)
{  
    // Determines edges vertex belongs to
    std::vector<Edge*> edgesA;
    edgesA.resize(inputVertices.size());
    std::vector<Edge*> edgesB;
    edgesB.resize(inputVertices.size());

    std::map<float, std::unordered_set<Edge*>> orderedEdges;
    for (size_t i = 0; i < inputVertices.size(); i += 3)
    {
        Edge* temp = new Edge;
        temp->length = DistanceSquared(inputVertices[i].pos, inputVertices[i + 1].pos);
        temp->vertices.first = i;
        temp->vertices.second = i + 1;
        orderedEdges[temp->length].insert(temp);
        edgesA[i] = temp;
        edgesB[i + 1] = temp;

        temp = new Edge;
        temp->length = DistanceSquared(inputVertices[i].pos, inputVertices[i + 2].pos);
        temp->vertices.first = i;
        temp->vertices.second = i + 2;
        orderedEdges[temp->length].insert(temp);
        edgesB[i] = temp;
        edgesA[i + 2] = temp;

        temp = new Edge;
        temp->length = DistanceSquared(inputVertices[i + 1].pos, inputVertices[i + 2].pos);
        temp->vertices.first = i + 1;
        temp->vertices.second = i + 2;
        orderedEdges[temp->length].insert(temp);
        edgesA[i + 1] = temp;
        edgesB[i + 2] = temp;
    }

    std::unordered_map<glm::vec3, std::unordered_set<size_t>> samePosition;
    for (size_t i = 0; i < inputVertices.size(); i++)
        samePosition[inputVertices[i].pos].insert(i);

    // Using squared length because it's faster
    length *= length;

    auto edge = orderedEdges.begin();
    while (edge != orderedEdges.end())
    {
        if (edge->first > length)
            break;

        Edge temp = *(*(edge->second.begin()));

        RemoveTriangle((temp.vertices.first / 3) * 3, orderedEdges, samePosition, inputVertices, edgesA);

        if (inputVertices[temp.vertices.first].pos != inputVertices[temp.vertices.second].pos)
        {
            std::unordered_set<Edge*> recalculateDistance;
            
            //----------------------------------------------Normal-----------------------------------------------------------
            std::unordered_set<size_t> recalculateNormalFirst = samePosition[inputVertices[temp.vertices.first].pos];
            std::unordered_set<size_t> recalculateNormalSecond = samePosition[inputVertices[temp.vertices.second].pos];
            //---------------------------------------------------------------------------------------------------------------

            glm::vec3 newPosition = (inputVertices[temp.vertices.first].pos + inputVertices[temp.vertices.second].pos) * 0.5f;
            
            //----------------------------------------------Normal-----------------------------------------------------------
            glm::vec3 newNormal = (inputVertices[temp.vertices.first].normal + inputVertices[temp.vertices.second].normal) * 0.5f;

            auto x = recalculateNormalFirst.begin();
            while (x != recalculateNormalFirst.end())
            {
                if (inputVertices[*x].normal == inputVertices[temp.vertices.first].normal)
                {
                    inputVertices[*x].normal = newNormal;
                    x = recalculateNormalFirst.erase(x);
                }
                else
                    x++;
            }

            x = recalculateNormalSecond.begin();
            while (x != recalculateNormalSecond.end())
            {
                if (inputVertices[*x].normal == inputVertices[temp.vertices.second].normal)
                {
                    inputVertices[*x].normal = newNormal;
                    x = recalculateNormalSecond.erase(x);
                }
                else
                    x++;
            }
            //---------------------------------------------------------------------------------------------------------------

            auto point = samePosition[inputVertices[temp.vertices.first].pos].begin();
            while (point != samePosition[inputVertices[temp.vertices.first].pos].end())
            {
                recalculateDistance.insert(edgesA[*point]);
                recalculateDistance.insert(edgesB[*point]);
                inputVertices[*point].pos = newPosition;
                //inputVertices[*point].normal = newNormal;
                samePosition[newPosition].insert(*point);
                point = samePosition[inputVertices[temp.vertices.first].pos].erase(point);
            }

            point = samePosition[inputVertices[temp.vertices.second].pos].begin();
            while (point != samePosition[inputVertices[temp.vertices.second].pos].end())
            {
                if (recalculateDistance.insert(edgesA[*point]).second && recalculateDistance.insert(edgesB[*point]).second)
                {
                    inputVertices[*point].pos = newPosition;
                    //inputVertices[*point].normal = newNormal;
                    samePosition[newPosition].insert(*point);
                    samePosition[inputVertices[temp.vertices.second].pos].erase(point);
                }
                else
                {
                    //----------------------------------------------Normal-----------------------------------------------------------
                    // Trying to find index of second vertex on the same edge
                    size_t firstIndex;
                    if (inputVertices[edgesA[*point]->vertices.first].pos == inputVertices[temp.vertices.first].pos)
                        firstIndex = edgesA[*point]->vertices.first;
                    else if (inputVertices[edgesA[*point]->vertices.second].pos == inputVertices[temp.vertices.first].pos)
                        firstIndex = edgesA[*point]->vertices.second;
                    else if (inputVertices[edgesB[*point]->vertices.first].pos == inputVertices[temp.vertices.first].pos)
                        firstIndex = edgesB[*point]->vertices.first;
                    else
                        firstIndex = edgesB[*point]->vertices.second;
                    //---------------------------------------------------------------------------------------------------------------
                    
                    size_t triangle = (*point) - (*point) % 3;
                    recalculateDistance.erase(edgesA[triangle]);
                    recalculateDistance.erase(edgesA[triangle + 1]);
                    recalculateDistance.erase(edgesA[triangle + 2]);
                    
                    //----------------------------------------------Normal-----------------------------------------------------------
                    recalculateNormalFirst.erase(triangle);
                    recalculateNormalFirst.erase(triangle + 1);
                    recalculateNormalFirst.erase(triangle + 2);
                    recalculateNormalSecond.erase(triangle);
                    recalculateNormalSecond.erase(triangle + 1);
                    recalculateNormalSecond.erase(triangle + 2);

                    newNormal = (inputVertices[firstIndex].normal + inputVertices[*point].normal) * 0.5f;

                    x = recalculateNormalFirst.begin();
                    while (x != recalculateNormalFirst.end())
                    {
                        if (inputVertices[*x].normal == inputVertices[firstIndex].normal)
                        {
                            inputVertices[*x].normal = newNormal;
                            x = recalculateNormalFirst.erase(x);
                        }
                        else
                            x++;
                    }

                    x = recalculateNormalSecond.begin();
                    while (x != recalculateNormalSecond.end())
                    {
                        if (inputVertices[*x].normal == inputVertices[*point].normal)
                        {
                            inputVertices[*x].normal = newNormal;
                            x = recalculateNormalSecond.erase(x);
                        }
                        else
                            x++;
                    }
                    //---------------------------------------------------------------------------------------------------------------

                    RemoveTriangle(triangle, orderedEdges, samePosition, inputVertices, edgesA);
                }

                point = samePosition[inputVertices[temp.vertices.second].pos].begin();
            }

            auto i = recalculateDistance.begin();
            while (i != recalculateDistance.end())
            {
                orderedEdges[(*i)->length].erase((*i));
                if (orderedEdges[(*i)->length].empty())
                    orderedEdges.erase((*i)->length);
                (*i)->length = DistanceSquared(inputVertices[(*i)->vertices.first].pos, inputVertices[(*i)->vertices.second].pos);
                orderedEdges[(*i)->length].insert((*i));

                i = recalculateDistance.erase(i);
            }
        }
        
        edge = orderedEdges.begin();
    }

    size_t insert = 0;
    for (size_t i = 0; i < inputVertices.size(); i++)
    {
        if (edgesA[i] != nullptr)
        {
            inputVertices[insert++] = inputVertices[i];
            delete edgesA[i];
        }
    }
    inputVertices.resize(insert);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------------------

//-------------------------------------------------------Vertex-clustering-------------------------------------------------------------------------------------
struct Box
{
    int64_t x;
    int64_t y;
    int64_t z;

    bool operator==(const Box& other) const
    {
        return x == other.x && y == other.y && z == other.z;
    }
};

template<> struct std::hash<Box> {
    size_t operator()(Box const& box) const
    {
        std::size_t res = 0;
        hash_combine(res, box.x);
        hash_combine(res, box.y);
        hash_combine(res, box.z);
        return res;
    }
};

// Determines if any vertex in a is in a same triangle as any vertex in b, vertices in a and b have the same position 
bool isConnected(std::vector<size_t>& a, std::vector<size_t>& b)
{
    for (size_t i = 0; i < a.size(); i++)
    {
        for (size_t x = 0; x < b.size(); x++)
        {
            if ((a[i] / 3) == (b[x] / 3))
                return true;
        }
    }

    return false;
}

void VertexClustering(std::vector<Vertex>& inputVertices, float length)
{
    std::unordered_map<glm::vec3, std::vector<size_t>> samePosition;
    for (size_t i = 0; i < inputVertices.size(); i++)
        samePosition[inputVertices[i].pos].push_back(i);

    //----------------------------------------------Normal/Color/Texture-----------------------------------------------------------
    std::vector<std::unordered_map<glm::vec3, std::vector<size_t>>> sameNormal;
    std::vector<std::unordered_map<glm::vec3, std::vector<size_t>>> sameColor;
    std::vector<std::unordered_map<glm::vec2, std::vector<size_t>>> sameTexture;
    for (auto i = samePosition.begin(); i != samePosition.end(); i++)
    {
        std::unordered_map<glm::vec3, std::vector<size_t>> tempNormal;
        std::unordered_map<glm::vec3, std::vector<size_t>> tempColor;
        std::unordered_map<glm::vec2, std::vector<size_t>> tempTexture;

        for (size_t x = 0; x < i->second.size(); x++)
        {
            tempNormal[inputVertices[i->second[x]].normal].push_back(i->second[x]);
            tempColor[inputVertices[i->second[x]].color].push_back(i->second[x]);
            tempTexture[inputVertices[i->second[x]].texCoord].push_back(i->second[x]);
        }

        sameNormal.push_back(tempNormal);
        sameColor.push_back(tempColor);
        sameTexture.push_back(tempTexture);
    }
    //-----------------------------------------------------------------------------------------------------------------------------

    length = 1 / length;

    std::unordered_map<Box, std::list<std::vector<size_t>>> clusters;
    for (auto i = samePosition.begin(); i != samePosition.end(); i++)
    {
        Box box = { static_cast<int64_t>(i->first.x * length), static_cast<int64_t>(i->first.y * length), static_cast<int64_t>(i->first.z * length) };
        clusters[box].push_back(i->second);
    }

    for (auto i = clusters.begin(); i != clusters.end(); i++)
    {
        for (auto x = ++(i->second.begin()); x != i->second.end(); x++)
            inputVertices[(i->second.front())[0]].pos += inputVertices[x->at(0)].pos;

        inputVertices[(i->second.front())[0]].pos /= i->second.size();

        for (auto x = ++(i->second.begin()); x != i->second.end(); x++)
            inputVertices[x->at(0)].pos = inputVertices[(i->second.front())[0]].pos;
    }

    for (auto i = samePosition.begin(); i != samePosition.end(); i++)
    {
        for (size_t x = 1; x < i->second.size(); x++)
            inputVertices[i->second[x]].pos = inputVertices[i->second[0]].pos;
    }

    /*//-----------------------------------------Normal-Average----------------------------------------------------------------------
    for (auto i = clusters.begin(); i != clusters.end(); i++)
    {
        for (auto x = ++(i->second.begin()); x != i->second.end(); x++)
            inputVertices[(i->second.front())[0]].normal += inputVertices[x->at(0)].normal;

        inputVertices[(i->second.front())[0]].normal /= i->second.size();

        for (auto x = ++(i->second.begin()); x != i->second.end(); x++)
            inputVertices[x->at(0)].normal = inputVertices[(i->second.front())[0]].normal;
    }

    for (auto i = samePosition.begin(); i != samePosition.end(); i++)
    {
        for (size_t x = 1; x < i->second.size(); x++)
            inputVertices[i->second[x]].normal = inputVertices[i->second[0]].normal;
    }
    //------------------------------------------------------------------------------------------------------------------------------*/

    //----------------------------------------------Normal-------------------------------------------------------------------------
    std::unordered_map<Box, std::list<std::vector<size_t>>> normalClusters;
    for (size_t i = 0; i < sameNormal.size(); i++)
    {
        for (auto x = sameNormal[i].begin(); x != sameNormal[i].end(); x++)
        {
            Box box = { static_cast<int64_t>(inputVertices[x->second[0]].pos.x * length), static_cast<int64_t>(inputVertices[x->second[0]].pos.y * length),
                static_cast<int64_t>(inputVertices[x->second[0]].pos.z * length) };
            normalClusters[box].push_back(x->second);
        }
    }

    std::vector<size_t> normalConnections;
    normalConnections.resize(inputVertices.size());

    // Sorting vertices into groups to later determine vertex normal should be average of all surviving vertices in same group
    for (auto i = normalClusters.begin(); i != normalClusters.end(); i++)
    {
        std::list<std::list<std::vector<size_t>>> connected;
        std::list<std::vector<size_t>> queue;

        while (!(i->second.empty()))
        {
            if (queue.empty())
            {
                queue.push_back(i->second.front());

                std::list<std::vector<size_t>> temp;

                temp.push_back(i->second.front());
                i->second.pop_front();

                connected.push_back(temp);
            }


            for (auto x = i->second.begin(); x != i->second.end(); x++)
            {
                if (isConnected(*x, queue.front()))
                {
                    connected.back().push_back(*x);
                    queue.push_back(*x);
                    x = i->second.erase(x);
                    x--;
                }
            }

            queue.pop_front();
        }

        int family = 0;
        for (auto x = connected.begin(); x != connected.end(); x++)
        {
            for (auto z = x->begin(); z != x->end(); z++)
            {
                for (size_t y = 0; y < z->size(); y++)
                    normalConnections[z->at(y)] = family;
            }

            family++;
        }
    }
    //-----------------------------------------------------------------------------------------------------------------------------

    //----------------------------------------------Color--------------------------------------------------------------------------
    std::unordered_map<Box, std::list<std::vector<size_t>>> colorClusters;
    for (size_t i = 0; i < sameColor.size(); i++)
    {
        for (auto x = sameColor[i].begin(); x != sameColor[i].end(); x++)
        {
            Box box = { static_cast<int64_t>(inputVertices[x->second[0]].pos.x * length), static_cast<int64_t>(inputVertices[x->second[0]].pos.y * length),
                static_cast<int64_t>(inputVertices[x->second[0]].pos.z * length) };
            colorClusters[box].push_back(x->second);
        }
    }

    std::vector<size_t> colorConnections;
    colorConnections.resize(inputVertices.size());

    // Sorting vertices into groups to later determine vertex color should be average of all surviving vertices in same group
    for (auto i = colorClusters.begin(); i != colorClusters.end(); i++)
    {
        std::list<std::list<std::vector<size_t>>> connected;
        std::list<std::vector<size_t>> queue;

        while (!(i->second.empty()))
        {
            if (queue.empty())
            {
                queue.push_back(i->second.front());

                std::list<std::vector<size_t>> temp;

                temp.push_back(i->second.front());
                i->second.pop_front();

                connected.push_back(temp);
            }


            for (auto x = i->second.begin(); x != i->second.end(); x++)
            {
                if (isConnected(*x, queue.front()))
                {
                    connected.back().push_back(*x);
                    queue.push_back(*x);
                    x = i->second.erase(x);
                    x--;
                }
            }

            queue.pop_front();
        }

        int family = 0;
        for (auto x = connected.begin(); x != connected.end(); x++)
        {
            for (auto z = x->begin(); z != x->end(); z++)
            {
                for (size_t y = 0; y < z->size(); y++)
                    colorConnections[z->at(y)] = family;
            }

            family++;
        }
    }
    //-----------------------------------------------------------------------------------------------------------------------------

    //----------------------------------------------Texture------------------------------------------------------------------------
    std::unordered_map<Box, std::list<std::vector<size_t>>> textureClusters;
    for (size_t i = 0; i < sameTexture.size(); i++)
    {
        for (auto x = sameTexture[i].begin(); x != sameTexture[i].end(); x++)
        {
            Box box = { static_cast<int64_t>(inputVertices[x->second[0]].pos.x * length), static_cast<int64_t>(inputVertices[x->second[0]].pos.y * length),
                static_cast<int64_t>(inputVertices[x->second[0]].pos.z * length) };
            textureClusters[box].push_back(x->second);
        }
    }

    std::vector<size_t> textureConnections;
    textureConnections.resize(inputVertices.size());

    // Sorting vertices into groups to later determine vertex texture should be average of all surviving vertices in same group
    for (auto i = textureClusters.begin(); i != textureClusters.end(); i++)
    {
        std::list<std::list<std::vector<size_t>>> connected;
        std::list<std::vector<size_t>> queue;

        while (!(i->second.empty()))
        {
            if (queue.empty())
            {
                queue.push_back(i->second.front());

                std::list<std::vector<size_t>> temp;

                temp.push_back(i->second.front());
                i->second.pop_front();

                connected.push_back(temp);
            }


            for (auto x = i->second.begin(); x != i->second.end(); x++)
            {
                if (isConnected(*x, queue.front()))
                {
                    connected.back().push_back(*x);
                    queue.push_back(*x);
                    x = i->second.erase(x);
                    x--;
                }
            }

            queue.pop_front();
        }

        int family = 0;
        for (auto x = connected.begin(); x != connected.end(); x++)
        {
            for (auto z = x->begin(); z != x->end(); z++)
            {
                for (size_t y = 0; y < z->size(); y++)
                    textureConnections[z->at(y)] = family;
            }

            family++;
        }
    }
    //-----------------------------------------------------------------------------------------------------------------------------

    //----------------------------------------------Normal/Color/Texture-----------------------------------------------------------
    // newClusters contains only triangles which have vertices in three different clusters
    std::unordered_map<Box, std::list<size_t>> newClusters;
    for (size_t i = 0; i < inputVertices.size(); i += 3)
    {
        if ((inputVertices[i].pos != inputVertices[i + 1].pos) && (inputVertices[i + 1].pos != inputVertices[i + 2].pos) && (inputVertices[i + 0].pos != inputVertices[i + 2].pos))
        {
            for (size_t x = 0; x < 3; x++)
            {
                Box box = { static_cast<int64_t>(inputVertices[i + x].pos.x * length), static_cast<int64_t>(inputVertices[i + x].pos.y * length),
                    static_cast<int64_t>(inputVertices[i + x].pos.z * length) };
                newClusters[box].push_back(i + x);
            }
        }
    }

    for (auto i = newClusters.begin(); i != newClusters.end(); i++)
    {
        std::list<size_t> inCluster = i->second;

        while (!(inCluster.empty()))
        {
            std::list<size_t> temp;
            temp.push_back(inCluster.front());
            inCluster.pop_front();

            for (auto x = inCluster.begin(); x != inCluster.end(); x++)
            {
                if (normalConnections[*x] == normalConnections[temp.front()])
                {
                    temp.push_back(*x);
                    x = inCluster.erase(x);
                    x--;
                }
            }

            for (auto x = ++(temp.begin()); x != temp.end(); x++)
                inputVertices[temp.front()].normal += inputVertices[*x].normal;

            inputVertices[temp.front()].normal /= temp.size();

            for (auto x = temp.begin(); x != temp.end(); x++)
                inputVertices[*x].normal = inputVertices[temp.front()].normal;
        }

        inCluster = i->second;

        while (!(inCluster.empty()))
        {
            std::list<size_t> temp;
            temp.push_back(inCluster.front());
            inCluster.pop_front();

            for (auto x = inCluster.begin(); x != inCluster.end(); x++)
            {
                if (colorConnections[*x] == normalConnections[temp.front()])
                {
                    temp.push_back(*x);
                    x = inCluster.erase(x);
                    x--;
                }
            }

            for (auto x = ++(temp.begin()); x != temp.end(); x++)
                inputVertices[temp.front()].color += inputVertices[*x].color;

            inputVertices[temp.front()].color /= temp.size();

            for (auto x = temp.begin(); x != temp.end(); x++)
                inputVertices[*x].color = inputVertices[temp.front()].color;
        }

        inCluster = i->second;

        while (!(inCluster.empty()))
        {
            std::list<size_t> temp;
            temp.push_back(inCluster.front());
            inCluster.pop_front();

            for (auto x = inCluster.begin(); x != inCluster.end(); x++)
            {
                if (textureConnections[*x] == textureConnections[temp.front()])
                {
                    temp.push_back(*x);
                    x = inCluster.erase(x);
                    x--;
                }
            }

            for (auto x = ++(temp.begin()); x != temp.end(); x++)
                inputVertices[temp.front()].texCoord += inputVertices[*x].texCoord;

            inputVertices[temp.front()].texCoord /= temp.size();

            for (auto x = temp.begin(); x != temp.end(); x++)
                inputVertices[*x].texCoord = inputVertices[temp.front()].texCoord;
        }
    }
    //-----------------------------------------------------------------------------------------------------------------------------

    size_t insert = 0;
    for (size_t i = 0; i < inputVertices.size(); i += 3)
    {
        if ((inputVertices[i].pos != inputVertices[i + 1].pos) && (inputVertices[i + 1].pos != inputVertices[i + 2].pos) && (inputVertices[i + 0].pos != inputVertices[i + 2].pos))
        {
            inputVertices[insert++] = inputVertices[i];
            inputVertices[insert++] = inputVertices[i + 1];
            inputVertices[insert++] = inputVertices[i + 2];
        }
    }
    inputVertices.resize(insert);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------------------