#pragma once
#include "CommonInclude.hpp"

#include "Mesh.hpp"

namespace pt {
	struct MaterialPrincipledBSDF {
		glm::vec3 baseColor;

		float Metalic;
		float Roughness;
		float IOR;
		float Alpha;
	};

	namespace ObjParser {
		pt::PtError parseFile(std::string fileName, std::vector<Mesh>& meshs);
		pt::PtError parseStream(std::istream& stream, std::vector<Mesh>& meshs);
	};

	namespace MtlParser {
		pt::PtError parseFile(std::string fileName, std::vector<Mesh>& meshs);
		pt::PtError parseStream(std::istream& stream, std::vector<Mesh>& meshs);
	};
}