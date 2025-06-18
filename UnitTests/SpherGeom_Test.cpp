#include "pch.h"
#include "CppUnitTest.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

#include "Renderer/OptiX/PBRT/Conductor.h"
#include "Renderer/OptiX/PBRT/GlossyDiffuse.h"
#include <iostream>

namespace PBRT_Tests
{
	TEST_CLASS(PBRT_Tests)
	{
	public:
		
		TEST_METHOD(CosTheta)
		{
			glm::vec3 vector(1, 2, 3);
			float result = PBRT::SpherGeom::CosTheta(vector);
			Assert::AreEqual(result, vector.z);
		}

		TEST_METHOD(EnergyConservation_Conductor)
		{
			unsigned int randomSeed = 15615615665;

			Surface surface;
			surface.albedo = glm::vec3(1.0f);
			surface.roughness = 0.0f;

			for (int i = 0; i < 10; i++) {
				glm::vec2 uo{ Random01(), Random01()};
				glm::vec3 wo = SampleUniformHemisphere(uo);

				surface.outgoingRay = wo;
				//std::cout << "x: " << woL.x << "y: " << woL.y << "z: " << woL.z << std::endl;

				const int nSamples = 16384;
				glm::vec3 lightOutput = glm::vec3(0.0f);
				for (int j = 0; j < nSamples; j++) {
					glm::vec3 newDirection;
					float pdf;
					glm::vec3 colorSample;
					bool sampleTest = PBRT::Conductor::Sample_f(randomSeed, surface, newDirection, pdf, colorSample);
					if (sampleTest) {
						lightOutput += colorSample * AbsDot(newDirection, glm::vec3(0, 0, 1)) / pdf;
					}
				}

				lightOutput /= nSamples;

				if (SaveMax(lightOutput) >= 1.01f) {
					std::cout << "Assert failed at iteration " << i << std::endl;
					PrintVector("wo", wo);
					PrintVector("lightOutput", lightOutput);

					Assert::Fail();
				}
			}
		}

		TEST_METHOD(EnergyConservation_GlossyDiffuse)
		{
			unsigned int randomSeed = 15615615665;

			Surface surface;
			surface.albedo = glm::vec3(1.0f);
			surface.roughness = 0.0f;

			for (int i = 0; i < 10; i++) {
				glm::vec2 uo{ Random01(), Random01() };
				glm::vec3 wo = SampleUniformHemisphere(uo);

				surface.outgoingRay = wo;
				//std::cout << "x: " << woL.x << "y: " << woL.y << "z: " << woL.z << std::endl;

				const int nSamples = 16384;
				glm::vec3 lightOutput = glm::vec3(0.0f);
				for (int j = 0; j < nSamples; j++) {
					BSDFSample sample;
					bool sampleTest = PBRT::GlossyDiffuse::Sample_f(randomSeed, surface, sample);
					if (sampleTest) {
						lightOutput += sample.color * AbsDot(sample.direction, glm::vec3(0, 0, 1)) / sample.pdf;
					}
				}

				lightOutput /= nSamples;
				PrintVector("lightOutput", lightOutput);
				if (SaveMax(lightOutput) >= 1.01f) {
					std::cout << "Assert failed at iteration " << i << std::endl;
					PrintVector("wo", wo);
					PrintVector("lightOutput", lightOutput);

					Assert::Fail();
				}
			}
		}

	private:
		glm::vec3 SampleUniformHemisphere(glm::vec2 u) {
			float z = u[0];
			float r = sqrt(1 - Sqr(z));
			float phi = 2 * 3.14159265359f * u[1];
			return { r * std::cos(phi), r * std::sin(phi), z };
		}

		float Random01() {
			return ((double)rand() / (RAND_MAX));
		}

		void PrintVector(const char* name, glm::vec3 vec) {
			std::cout << name << ": " << vec.x << ", " << vec.y << ", " << vec.z << std::endl;
		}
	};
}
