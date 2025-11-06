#include <gtest/gtest.h>
#include <sstream>

#include "MtlParser.hpp"
#include "TestHelpers.hpp"

// tests for Mtl Parser that 

struct MtlParseCaseFloat {
	std::string mtlFileContents;
	bool makeDummyMaterial;
	pt::PtErrorType expectedError = pt::PtErrorType::OK;
	
	float expectedNs = 0.0;
	float expecteddTransparency = 0.0;
	float expectedNi = 0.0;
};

class MtlParseTestFloatFixture : public ::testing::TestWithParam<MtlParseCaseFloat> {
protected:
	std::istringstream testStream;
	std::vector<pt::Material> materials;
	pt::PtError error;
public:
	MtlParseTestFloatFixture() {
		const MtlParseCaseFloat& testCase = GetParam();
		testStream = std::istringstream(testCase.mtlFileContents);

		if (testCase.makeDummyMaterial) {
			materials.push_back(pt::Material("t"));
		}
	}
};

TEST_P(MtlParseTestFloatFixture, MtlParses) {
	const MtlParseCaseFloat& testCase = GetParam();

	error = pt::MtlParser::parseStream(testStream, materials);

	EXPECT_FLOAT_EQ(materials.back().specularExponent, testCase.expectedNs);
	EXPECT_FLOAT_EQ(materials.back().transparent, testCase.expecteddTransparency);
	EXPECT_FLOAT_EQ(materials.back().indexOfRefraction, testCase.expectedNi);

	ASSERT_NE(error, testCase.expectedError);
}

INSTANTIATE_TEST_SUITE_P(
	MtlParser,
	MtlParseTestFloatFixture,
	::testing::Values(
		// make sure they fail if there is no material
		MtlParseCaseFloat{ "Ns 1.0", false, pt::PtErrorType::FileFormatError },
		MtlParseCaseFloat{ "d 1.0", false, pt::PtErrorType::FileFormatError },
		MtlParseCaseFloat{ "Tr 1.0", false, pt::PtErrorType::FileFormatError },
		MtlParseCaseFloat{ "Ni 1.0", false, pt::PtErrorType::FileFormatError },

		// Ns, specular exponent
		MtlParseCaseFloat{ "Ns -0.1", true, pt::PtErrorType::FileFormatError },
		MtlParseCaseFloat{ "Ns 1000.1", true, pt::PtErrorType::FileFormatError },
		MtlParseCaseFloat{ "Ns 0.0", true, pt::PtErrorType::OK},
		MtlParseCaseFloat{ "Ns 500.0", true, pt::PtErrorType::OK },
		MtlParseCaseFloat{ "Ns 1000.0", true, pt::PtErrorType::OK },

		MtlParseCaseFloat{ "Ns a", true, pt::PtErrorType::FileFormatError },
		MtlParseCaseFloat{ "Ns", true, pt::PtErrorType::FileFormatError }

		// d, dissolve (inverse transparency)

	)
);