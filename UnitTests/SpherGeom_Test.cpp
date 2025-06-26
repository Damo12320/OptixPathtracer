#include "pch.h"
#include "CppUnitTest.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

#include "Renderer/OptiX/PBRT/Conductor.h"
#include "Renderer/OptiX/PBRT/GlossyDiffuse.h"
#include <iostream>
#include <functional>

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

#pragma region EnergyConservation

#pragma region Conductor

		TEST_METHOD(EnergyConservation_Conductor0R)
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
					bool sampleTest = PBRT::Conductor::Sample_f(randomSeed, surface, sample);
					if (sampleTest) {
						lightOutput += sample.color * AbsDot(sample.direction, glm::vec3(0, 0, 1)) / sample.pdf;
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

		TEST_METHOD(EnergyConservation_Conductor05R)
		{
			unsigned int randomSeed = 15615615665;

			Surface surface;
			surface.albedo = glm::vec3(1.0f);
			surface.roughness = 0.5f;

			for (int i = 0; i < 10; i++) {
				glm::vec2 uo{ Random01(), Random01() };
				glm::vec3 wo = SampleUniformHemisphere(uo);

				surface.outgoingRay = wo;
				//std::cout << "x: " << woL.x << "y: " << woL.y << "z: " << woL.z << std::endl;

				const int nSamples = 16384;
				glm::vec3 lightOutput = glm::vec3(0.0f);
				for (int j = 0; j < nSamples; j++) {
					BSDFSample sample;
					bool sampleTest = PBRT::Conductor::Sample_f(randomSeed, surface, sample);
					if (sampleTest) {
						lightOutput += sample.color * AbsDot(sample.direction, glm::vec3(0, 0, 1)) / sample.pdf;
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

		TEST_METHOD(EnergyConservation_Conductor1R)
		{
			unsigned int randomSeed = 15615615665;

			Surface surface;
			surface.albedo = glm::vec3(1.0f);
			surface.roughness = 1.0f;

			for (int i = 0; i < 10; i++) {
				glm::vec2 uo{ Random01(), Random01() };
				glm::vec3 wo = SampleUniformHemisphere(uo);

				surface.outgoingRay = wo;
				//std::cout << "x: " << woL.x << "y: " << woL.y << "z: " << woL.z << std::endl;

				const int nSamples = 16384;
				glm::vec3 lightOutput = glm::vec3(0.0f);
				for (int j = 0; j < nSamples; j++) {
					BSDFSample sample;
					bool sampleTest = PBRT::Conductor::Sample_f(randomSeed, surface, sample);
					if (sampleTest) {
						lightOutput += sample.color * AbsDot(sample.direction, glm::vec3(0, 0, 1)) / sample.pdf;
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

#pragma endregion

#pragma region GlossyDiffuse

		TEST_METHOD(EnergyConservation_GlossyDiffuse0R)
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

		TEST_METHOD(EnergyConservation_GlossyDiffuse05R)
		{
			unsigned int randomSeed = 15615615665;

			Surface surface;
			surface.albedo = glm::vec3(1.0f);
			surface.roughness = 0.5f;

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

		TEST_METHOD(EnergyConservation_GlossyDiffuse1R)
		{
			unsigned int randomSeed = 15615615665;

			Surface surface;
			surface.albedo = glm::vec3(1.0f);
			surface.roughness = 1.0f;

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

#pragma endregion

#pragma endregion

		/*TEST_METHOD(TestBRDF)
		{
			const int thetaRes = 80;
			const int phiRes = (2 * thetaRes);
			const int sampleCount = 1000000;
			const int chi2_Runs = 5;


			float* frequencies = new float[thetaRes * phiRes];
			float* expFrequencies = new float[thetaRes * phiRes];

			unsigned int randomSeed = 15615615665;

			Surface surface;
			surface.albedo = glm::vec3(1.0f);
			surface.roughness = 0.0f;

			auto bsdfFunction = [](unsigned int& rSeed, Surface& surface, BSDFSample& sample) {
				return PBRT::GlossyDiffuse::Sample_f(rSeed, surface, sample);
				};

			auto bsdf_PDFFunction = [](unsigned int& rSeed, Surface& surface, glm::vec3 wi) {
				return PBRT::GlossyDiffuse::PDF(surface.outgoingRay, );
				};

			for (int k = 0; k < chi2_Runs; k++) {
				glm::vec2 uo{ Random01(), Random01() };
				glm::vec3 wo = SampleUniformHemisphere(uo);
				surface.outgoingRay = wo;

				FrequencyTable(bsdfFunction, randomSeed, surface, sampleCount, thetaRes, phiRes, frequencies);
			}


			glm::vec3 vector(1, 2, 3);
			float result = PBRT::SpherGeom::CosTheta(vector);
			Assert::AreEqual(result, vector.z);

			delete[] frequencies;
			delete[] expFrequencies;
		}*/


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





		
		float AdaptiveSimpson(const std::function<float(float)>& f, float x0, float x1, float eps = 1e-6f, int depth = 6) {
			int count = 0;
			/* Define an recursive lambda function for integration over subintervals */
			std::function<float(float, float, float, float, float, float, float, float, int)>
				integrate = [&](float a, float b, float c, float fa, float fb, float fc, float I,
					float eps, int depth) {
						/* Evaluate the function at two intermediate points */
						float d = 0.5f * (a + b), e = 0.5f * (b + c), fd = f(d), fe = f(e);

						/* Simpson integration over each subinterval */
						float h = c - a, I0 = (float)(1.0 / 12.0) * h * (fa + 4 * fd + fb),
							I1 = (float)(1.0 / 12.0) * h * (fb + 4 * fe + fc), Ip = I0 + I1;
						++count;

						/* Stopping criterion from J.N. Lyness (1969)
						  "Notes on the adaptive Simpson quadrature routine" */
						if (depth <= 0 || std::abs(Ip - I) < 15 * eps) {
							// Richardson extrapolation
							return Ip + (float)(1.0 / 15.0) * (Ip - I);
						}

						return integrate(a, d, b, fa, fd, fb, I0, .5f * eps, depth - 1) +
							integrate(b, e, c, fb, fe, fc, I1, .5f * eps, depth - 1);
				};
			float a = x0, b = 0.5f * (x0 + x1), c = x1;
			float fa = f(a), fb = f(b), fc = f(c);
			float I = (c - a) * (float)(1.0 / 6.0) * (fa + 4 * fb + fc);
			return integrate(a, b, c, fa, fb, fc, I, eps, depth);
		}

		float AdaptiveSimpson2D(const std::function<float(float, float)>& f, float x0, float y0, float x1, float y1, float eps = 1e-6f, int depth = 6) {
			/* Lambda function that integrates over the X axis */
			auto integrate = [&](float y) {
				return AdaptiveSimpson(std::bind(f, std::placeholders::_1, y), x0, x1, eps,
					depth);
				};
			float value = AdaptiveSimpson(integrate, y0, y1, eps, depth);
			return value;
		}

		void FrequencyTable(std::function<bool (unsigned int&, Surface&, BSDFSample&)> bsdf, unsigned int& randomSeed, Surface& surface, int sampleCount, int thetaRes, int phiRes, float* target) {
			memset(target, 0, thetaRes * phiRes * sizeof(float));

			float factorTheta = thetaRes / 3.14159265359f, factorPhi = phiRes / (2.0f * 3.14159265359f);

			glm::vec3 wi;
			for (int i = 0; i < sampleCount; i++) {
				BSDFSample sample;
				bool sampleTest = bsdf(randomSeed, surface, sample);
				if (!sampleTest || sample.specular) {
					continue;
				}

				glm::vec2 coords(acos(sample.direction.z) * factorTheta,
					std::atan2(sample.direction.y, sample.direction.x) * factorPhi);

				if (coords.y < 0.0f)
					coords.y += 2.0f * 3.14159265359f * factorPhi;

				int thetaBin = std::min(std::max(0, (int)std::floor(coords.x)), thetaRes - 1);
				int phiBin = std::min(std::max(0, (int)std::floor(coords.y)), phiRes - 1);

				target[thetaBin * phiRes + phiBin] += 1;
			}
		}

		void IntegrateFrequencyTable(std::function<float(unsigned int&, Surface&)> bsdf_PDF, unsigned int& randomSeed, Surface& surface, int sampleCount, int thetaRes, int phiRes, float* target) {
			memset(target, 0, thetaRes * phiRes * sizeof(float));

			float factorTheta = 3.14159265359f / thetaRes, factorPhi = (2 * 3.14159265359f) / phiRes;

			for (int i = 0; i < thetaRes; ++i) {
				for (int j = 0; j < phiRes; ++j) {
					*target++ =
						sampleCount *
						AdaptiveSimpson2D(
							[&](float theta, float phi) -> float {
								float cosTheta = std::cos(theta), sinTheta = std::sin(theta);
								float cosPhi = std::cos(phi), sinPhi = std::sin(phi);
								glm::vec3 wiL(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
								return bsdf_PDF(randomSeed, surface) * sinTheta;
							},
							i * factorTheta, j * factorPhi, (i + 1) * factorTheta,
							(j + 1) * factorPhi);
				}
			}
		}
	};
}
