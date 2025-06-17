#include "WriteImage.h"

#include "../../3rdParty/stb_image_write.h"

namespace WriteImage {
	bool WriteBMP(std::vector<uint64_t>* pixels, glm::ivec2 size, const char* path) {
		std::vector<uint8_t> imageData;
		imageData.reserve(size.x * size.y * 4);

		for (uint64_t pixel: *pixels) {
			uint16_t r16 = pixel			& 0xFFFF;
			uint16_t g16 = (pixel >> 16)	& 0xFFFF;
			uint16_t b16 = (pixel >> 32)	& 0xFFFF;
			uint16_t a16 = (pixel >> 48)	& 0xFFFF;

			uint8_t r8 = pixel >> 8;
			uint8_t g8 = pixel >> 8;
			uint8_t b8 = pixel >> 8;
			uint8_t a8 = pixel >> 8;

			imageData.push_back(r8);
			imageData.push_back(g8);
			imageData.push_back(b8);
			imageData.push_back(a8);
		}

		stbi_write_bmp(path, size.x, size.y, 4, pixels->data());

		return true;
	}
}