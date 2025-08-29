#include <gtest/gtest.h>
#include <sstream>

#include "ObjParser.hpp"
#include "TestHelpers.hpp"

// Parameterized testing help from here:
// https://www.sandordargo.com/blog/2019/04/24/parameterized-testing-with-gtest

struct ObjPasrseGeneralCase {
	std::string objFileContents;
	bool shouldBeOk;
	bool needsDummyMesh;
};

class ObjParserGeneralTestFixture : public ::testing::TestWithParam<ObjPasrseGeneralCase> {
protected:
	std::istringstream testStream;
	std::vector<pt::Mesh> meshs;
	pt::PtError error;
public:
	ObjParserGeneralTestFixture() {
		const ObjPasrseGeneralCase& testCase = GetParam();
		testStream = std::istringstream(testCase.objFileContents);

		if (testCase.needsDummyMesh) {
			meshs.push_back(pt::Mesh("t"));
		}
	}
};

TEST_P(ObjParserGeneralTestFixture, ParsesVertex) {
	const ObjPasrseGeneralCase& testCase = GetParam();
	error = pt::ObjParser::parseStream(testStream, meshs);

	if (testCase.shouldBeOk) {
		EXPECT_EQ(error, pt::PtErrorType::OK);
	} else {
		EXPECT_NE(error, pt::PtErrorType::OK);
	}
	
}

INSTANTIATE_TEST_CASE_P(
	ParsesVertexVariousCases,
	ObjParserGeneralTestFixture,
	::testing::Values(
		ObjPasrseGeneralCase{ "v", false, true },							// rejects no values
		ObjPasrseGeneralCase{ "v 1.0", false, true },						// rejects one value
		ObjPasrseGeneralCase{ "v 1.0 1.0", false, true },					// rejects two values
		ObjPasrseGeneralCase{ "v 1.0 1.0 1.0", false, true },				// accepts three values
		ObjPasrseGeneralCase{ "v 1.0 1.0 1.0 1.0", false, true },			// accepts four values (NOTE: any value past 3 is ignored)
		ObjPasrseGeneralCase{ "v 1.0 1.0 1.0 1.0 1.0", false, true },		// accepts five (NOTE: any value past 3 is ignored)
		ObjPasrseGeneralCase{ "v 1.0 1.0 1.0 1.0 1.0 1.0", false, true }	// accepts six (NOTE: any value past 3 is ignored)
	)
);