#include "ObjParser.hpp"
#include <cctype>

pt::PtError pt::ObjParser::parseFile(std::string fileName, std::vector<Mesh>& meshs) {
	std::ifstream inFS(fileName);

	pt::PtError error = parseStream(inFS, meshs);
	
	if (error != pt::PtErrorType::OK) {
		std::cout << "Reading OBJ File Failed - '" << error.message << "'" << std::endl;
	}

	return error;
}

pt::PtError pt::ObjParser::parseStream(std::istream& stream, std::vector<Mesh>& meshs) {
	std::string line;
	getline(stream, line);

	std::istringstream lineStream(line);

	while (!stream.fail()) {
		std::string elementType;
		lineStream >> elementType;

		if (elementType == "o") {
			// new object
			std::string name;
			lineStream >> name;
			meshs.emplace_back(name);

		} else if (elementType == "v") {
			// vertex
			// we will ignore w
			float x = 0, y = 0, z = 0;
			if (!(lineStream >> x >> y >> z)) {
				return pt::PtError(pt::PtErrorType::FileFormatError, "Reading in a vertex failed");
			}
			meshs.back().vertices.emplace_back(x, y, z);

		} else if (elementType == "vn") {
			// vertex normal
			float x = 0, y = 0, z = 0;
			lineStream >> x >> y >> z;

			// its not necessarily normalized, so normalize it to make sure
			glm::vec3 vec(x, y, z);
			vec = glm::normalize(vec);

			meshs.back().vertexNormals.emplace_back(vec.x, vec.y, vec.z);

		} else if (elementType == "vt") {
			// vertex texture
			// last two are optional, but default to zero so this should be fine
			float x = 0, y = 0, z = 0;
			lineStream >> x >> y >> z;
			meshs.back().vertexTextureCoordinates.emplace_back(x, y, z);

		} else if (elementType == "f") {
			// face
			// f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
			std::array<std::string, 3> faces;

			lineStream >> faces[0] >> faces[1] >> faces[2];

			for (std::string f : faces) {
				int v = -1, vt = -1, vn = -1; // -1 means does not have

				int firstSlashIndex = f.find_first_of('/');
				int secondSlashIndex = f.find_last_of('/');

				if (firstSlashIndex == std::string::npos) {
					// no slashes
					v = stol(f);
				} else if (firstSlashIndex == secondSlashIndex - 1) {
					// no vt
					v = stol(f.substr(0, firstSlashIndex));
					vn = stol(f.substr(secondSlashIndex + 1));
				} else {
					v = stol(f.substr(0, firstSlashIndex));
					vt = stol(f.substr(firstSlashIndex + 1, f.size() - firstSlashIndex - 1));
					vn = stol(f.substr(secondSlashIndex + 1));
				}
			}
		} else if (elementType == "usemtl") {

		} else if (elementType == "mtllib") {

		} else {
			std::cout << "Dont know what " << elementType << " means";
		}

		getline(stream, line);
		lineStream = std::istringstream(line);

		return pt::PtErrorType::OK;
	}
}