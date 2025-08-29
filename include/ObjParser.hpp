#pragma once
#include "CommonInclude.hpp"

#include <fstream>
#include <sstream>

namespace pt {
	struct MaterialPrincipledBSDF {
		glm::vec3 baseColor;

		float Metalic;
		float Roughness;
		float IOR;
		float Alpha;
	};

	struct Mesh {
		std::vector<glm::vec3> vertices;
		std::vector<glm::vec3> vertexTextureCoordinates;
		std::vector<glm::vec3> vertexNormals;

		std::vector<int> vertexIndexes;
		std::vector<int> vertexTextureCoordinatesIndexes;
		std::vector<int> vertexNormalIndexes;

		std::string name;

		Mesh(std::string name) : name(name) {}
	};

	namespace ObjParser {
		pt::PtError parseFile(std::string fileName, std::vector<Mesh>& meshs);
		pt::PtError parseStream(std::istream& stream, std::vector<Mesh>& meshs);
	};

	namespace MtlParser {

	};
}