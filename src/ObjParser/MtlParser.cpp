#include "CommonInclude.hpp"
#include "MtlParser.hpp"

namespace MtlParserHelpers {
	static pt::PtError ensureMaterialExists(const std::vector<pt::Material>& materials) {
		if (materials.size() == 0) {
			return pt::PtError(pt::PtErrorType::FileFormatError, "Trying to read data before any meshs have been defined");
		}
		return pt::PtErrorType::OK;
	}

	static pt::PtError newMaterial(std::istream& lineStream, std::vector<pt::Material>& materials) {
		std::string materialName;
		lineStream >> materialName;

		materials.emplace_back(materialName);

		return pt::PtErrorType::OK;
	}

	static pt::PtError setAmbient(std::istream& lineStream, std::vector<pt::Material>& materials) {
		float x, y, z;

		if (!(lineStream >> x >> y >> z)) {
			return pt::PtError(pt::PtErrorType::FileFormatError, "Reading in ambient failed");
		}

		materials.back().ambientColor = glm::vec3(x, y, z);

		return pt::PtErrorType::OK;
	}

	static pt::PtError setDiffuse(std::istream& lineStream, std::vector<pt::Material>& materials) {
		float x, y, z;

		if (!(lineStream >> x >> y >> z)) {
			return pt::PtError(pt::PtErrorType::FileFormatError, "Reading in diffuse failed");
		}

		materials.back().diffuseColor = glm::vec3(x, y, z);

		return pt::PtErrorType::OK;
	}

	static pt::PtError setSpecular(std::istream& lineStream, std::vector<pt::Material>& materials) {
		float x, y, z;

		if (!(lineStream >> x >> y >> z)) {
			return pt::PtError(pt::PtErrorType::FileFormatError, "Reading in specular failed");
		}

		materials.back().specularColor = glm::vec3(x, y, z);

		return pt::PtErrorType::OK;
	}

	static pt::PtError setSpecularExponent(std::istream& lineStream, std::vector<pt::Material>& materials) {
		float x;

		if (!(lineStream >> x)) {
			return pt::PtError(pt::PtErrorType::FileFormatError, "Reading in specular exponent failed");
		}

		materials.back().specularExponent = x;

		return pt::PtErrorType::OK;
	}

	static pt::PtError setTransparent(std::istream& lineStream, std::vector<pt::Material>& materials) {
		float x;

		if (!(lineStream >> x)) {
			return pt::PtError(pt::PtErrorType::FileFormatError, "Reading in transparent failed");
		}

		materials.back().transparent = x;

		return pt::PtErrorType::OK;
	}

	static pt::PtError setInverseTransparent(std::istream& lineStream, std::vector<pt::Material>& materials) {
		float x;

		if (!(lineStream >> x)) {
			return pt::PtError(pt::PtErrorType::FileFormatError, "Reading in transparent failed");
		}

		materials.back().transparent = 1 - x;

		return pt::PtErrorType::OK;
	}

	static pt::PtError setTransmissionFilter(std::istream& lineStream, std::vector<pt::Material>& materials) {
		float x, y, z;

		if (!(lineStream >> x >> y >> z)) {
			return pt::PtError(pt::PtErrorType::FileFormatError, "Reading in transmission filter failed");
		}

		materials.back().transmissionFilter = glm::vec3(x, y, z);

		return pt::PtErrorType::OK;
	}

	static pt::PtError setIndexRefraction(std::istream& lineStream, std::vector<pt::Material>& materials) {
		float x;

		if (!(lineStream >> x)) {
			return pt::PtError(pt::PtErrorType::FileFormatError, "Reading in optical density/index of refraction failed");
		}

		materials.back().indexOfRefraction = x;

		return pt::PtErrorType::OK;
	}
}

pt::PtError pt::MtlParser::parseFile(std::string fileName, std::vector<pt::Material>& materials) {
	std::ifstream inFS(fileName);

	if (!inFS.is_open() || !inFS.good()) {
		std::ostringstream errorStream;
		errorStream << "error reading material file '" << fileName << "'";
		return pt::PtError(pt::PtErrorType::FileFormatError, errorStream.str());
	}

	pt::PtError error = pt::MtlParser::parseStream(inFS, materials);

	return error;
}

pt::PtError pt::MtlParser::parseStream(std::istream& stream, std::vector<pt::Material>& materials) {
	std::string line;
	std::getline(stream, line);

	std::istringstream lineStream(line);

	while (!stream.fail()) {
		std::string prefix;
		lineStream >> prefix;

		if (prefix == "Ka") {
			pt::PtError error = MtlParserHelpers::ensureMaterialExists(materials);

			if (error != pt::PtErrorType::OK) {
				return error;
			}

			error = MtlParserHelpers::setAmbient(lineStream, materials);

			if (error != pt::PtErrorType::OK) {
				return error;
			}
		} else if (prefix == "Kd") {
			pt::PtError error = MtlParserHelpers::ensureMaterialExists(materials);

			if (error != pt::PtErrorType::OK) {
				return error;
			}

			error = MtlParserHelpers::setDiffuse(lineStream, materials);

			if (error != pt::PtErrorType::OK) {
				return error;
			}

		} else if (prefix == "Ks") {
			pt::PtError error = MtlParserHelpers::ensureMaterialExists(materials);

			if (error != pt::PtErrorType::OK) {
				return error;
			}

			error = MtlParserHelpers::setSpecular(lineStream, materials);

			if (error != pt::PtErrorType::OK) {
				return error;
			}

		} else if (prefix == "Ns") {
			pt::PtError error = MtlParserHelpers::ensureMaterialExists(materials);

			if (error != pt::PtErrorType::OK) {
				return error;
			}

			error = MtlParserHelpers::setSpecularExponent(lineStream, materials);

			if (error != pt::PtErrorType::OK) {
				return error;
			}

		} else if (prefix == "d") {
			pt::PtError error = MtlParserHelpers::ensureMaterialExists(materials);

			if (error != pt::PtErrorType::OK) {
				return error;
			}

			error = MtlParserHelpers::setInverseTransparent(lineStream, materials);

			if (error != pt::PtErrorType::OK) {
				return error;
			}

		} else if (prefix == "Tr") {
			pt::PtError error = MtlParserHelpers::ensureMaterialExists(materials);

			if (error != pt::PtErrorType::OK) {
				return error;
			}

			error = MtlParserHelpers::setTransparent(lineStream, materials);

			if (error != pt::PtErrorType::OK) {
				return error;
			}

		} else if (prefix == "Tf") {
			pt::PtError error = MtlParserHelpers::ensureMaterialExists(materials);

			if (error != pt::PtErrorType::OK) {
				return error;
			}

			error = MtlParserHelpers::setTransmissionFilter(lineStream, materials);

			if (error != pt::PtErrorType::OK) {
				return error;
			}

		} else if (prefix == "Ni") {

			pt::PtError error = MtlParserHelpers::ensureMaterialExists(materials);

			if (error != pt::PtErrorType::OK) {
				return error;
			}

			error = MtlParserHelpers::setIndexRefraction(lineStream, materials);

			if (error != pt::PtErrorType::OK) {
				return error;
			}

		} else if (prefix == "map_Ka") {

		} else if (prefix == "map_Kd") {

		} else if (prefix == "map_Ks") {

		} else if (prefix == "map_Ns") {

		} else if (prefix == "map_d") {

		} else if (prefix == "map_bump") {

		} else if (prefix == "bump") {

		} else if (prefix == "disp") {

		} else if (prefix == "decal") {

		} else if (prefix == "illum") {

		} else if (prefix == "newmtl") {
			MtlParserHelpers::newMaterial(lineStream, materials);
		}
		
		getline(stream, line);
		lineStream = std::istringstream(line);
	}

	return pt::PtErrorType::OK;
}