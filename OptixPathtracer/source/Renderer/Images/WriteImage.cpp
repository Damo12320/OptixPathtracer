#include "WriteImage.h"

#include "../../3rdParty/stb_image_write.h"

namespace WriteImage {
	bool WriteBMP(std::vector<glm::vec3>& pixels, glm::ivec2 size, const char* path) {
		std::vector<uint8_t> imageData;
		imageData.reserve(size.x * size.y * 3);


		for (int y = size.y - 1; y >= 0; --y) {
			for (int x = 0; x < size.x; ++x) {
				const glm::vec3& pixel = pixels[y * size.x + x];

				uint8_t r = static_cast<uint8_t>(glm::clamp(pixel.r, 0.0f, 1.0f) * 255.0f);
				uint8_t g = static_cast<uint8_t>(glm::clamp(pixel.g, 0.0f, 1.0f) * 255.0f);
				uint8_t b = static_cast<uint8_t>(glm::clamp(pixel.b, 0.0f, 1.0f) * 255.0f);

				imageData.push_back(r);
				imageData.push_back(g);
				imageData.push_back(b);
			}
		}

		stbi_write_bmp(path, size.x, size.y, 3, imageData.data());

		std::cout << "Pixels written to: " << path << std::endl;

		return true;
	}

	void WriteTextureToBMP(GLTexture2D* texture, const char* path) {
		std::vector<glm::vec3> pixels;
		texture->DownloadTexture(pixels);

		glm::ivec2 size{ texture->GetWidth(), texture->GetHeight()};

		WriteBMP(pixels, size, path);
	}
}