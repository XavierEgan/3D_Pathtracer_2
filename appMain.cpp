#include <MtlParser.hpp>
#include <ObjParser.hpp>
#include <render.hpp>

#include <fstream>
#include <filesystem>
#include <iostream>
#include <iomanip>



int main() {
	std::vector<pt::Mesh> meshs;
	std::vector<pt::Material> materials;

	pt::Mesh dummyMesh("test");
    dummyMesh.vertices = {
        {0.0f, 0.0f, 1.0f},  // vertex 0
        {1.0f, 0.0f, 1.0f},  // vertex 1
        {1.0f, 1.0f, 1.0f},  // vertex 2
        {0.0f, 1.0f, 1.0f}   // vertex 3
    };
    // two triangles for a quad
    dummyMesh.vertexIndexes = {
        0, 1, 2,  // first triangle
        0, 2, 3   // second triangle
    };
    meshs.push_back(dummyMesh);


	render(meshs, materials);
}