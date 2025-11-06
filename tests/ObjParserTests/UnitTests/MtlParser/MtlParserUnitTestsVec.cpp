#include <gtest/gtest.h>
#include <sstream>

#include "MtlParser.hpp"
#include "TestHelpers.hpp"

// tests for Mtl Parser that 

struct MtlParseCaseVec {
	std::string mtlFileContents;
	bool makeDummyMaterial;
	pt::PtErrorType expectedError = pt::PtErrorType::OK;
	
};

class MtlParseTestVecFixture : public ::testing::TestWithParam<MtlParseCaseVec> {
protected:
	std::istringstream testStream;
	std::vector<pt::Material> materials;
	pt::PtError error;
public:
	MtlParseTestVecFixture() {
		const MtlParseCaseVec& testCase = GetParam();
		testStream = std::istringstream(testCase.mtlFileContents);

		if (testCase.makeDummyMaterial) {
			materials.push_back(pt::Material("t"));
		}
	}
};

TEST_P(MtlParseTestVecFixture, MtlParses) {
	const MtlParseCaseVec& testCase = GetParam();

	error = pt::MtlParser::parseStream(testStream, materials);

	ASSERT_NE(error, testCase.expectedError);


}

INSTANTIATE_TEST_SUITE_P(
	MtlParser,
	MtlParseTestVecFixture,
	::testing::Values(
		
	)
);