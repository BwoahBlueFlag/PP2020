// Project Practice 2020/2021 - Fast 3D scene rendering using LOD
// simply.h - api for mesh simplification methods 
// Author: Marek Janciar

#pragma once

#include"data.h"
#include<vector>

void LengthIncremental(std::vector<Vertex>& inputVertices, float length);
void VertexClustering(std::vector<Vertex>& inputVertices, float length);