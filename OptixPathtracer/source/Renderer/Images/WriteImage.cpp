#include "WriteImage.h"

#define TINYEXR_IMPLEMENTATION
#include "../../3rdParty/tinyexr.h"
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

	//https://github.com/syoyo/tinyexr/blob/release/examples/rgbe2exr/rgbe2exr.cc
	void WriteEXR(std::vector<glm::vec3>& pixels, glm::ivec2 size, const char* path) {
		EXRHeader header;
		InitEXRHeader(&header);

		EXRImage image;
		InitEXRImage(&image);
		image.num_channels = 3;

		std::vector<float> images[3];
		images[0].reserve(size.x * size.y);//R
		images[1].reserve(size.x * size.y);//G
		images[2].reserve(size.x * size.y);//B

		for (const glm::vec3& pixel : pixels) {
			images[0].push_back(pixel.r);//R
			images[1].push_back(pixel.g);//G
			images[2].push_back(pixel.b);//B
		}

		float* image_ptr[3];
		image_ptr[0] = &(images[2].at(0)); // B
		image_ptr[1] = &(images[1].at(0)); // G
		image_ptr[2] = &(images[0].at(0)); // R

		image.images = (unsigned char**)image_ptr;
		image.width = size.x;
		image.height = size.y;

		header.num_channels = 3;
		header.channels = (EXRChannelInfo*)malloc(sizeof(EXRChannelInfo) * header.num_channels);
		// Must be BGR(A) order, since most of EXR viewers expect this channel order.
		strncpy(header.channels[0].name, "B", 255); header.channels[0].name[strlen("B")] = '\0';
		strncpy(header.channels[1].name, "G", 255); header.channels[1].name[strlen("G")] = '\0';
		strncpy(header.channels[2].name, "R", 255); header.channels[2].name[strlen("R")] = '\0';


		header.pixel_types = (int*)malloc(sizeof(int) * header.num_channels);
		header.requested_pixel_types = (int*)malloc(sizeof(int) * header.num_channels);
		for (int i = 0; i < header.num_channels; i++) {
			header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT; // pixel type of input image
			header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT; // pixel type of output image to be stored in .EXR
		}

		const char* err;
		int ret = SaveEXRImageToFile(&image, &header, path, &err);
		if (ret != TINYEXR_SUCCESS) {
			std::cout << "EXR Save Error: " << err << std::endl;
			return;
		}

		free(header.channels);
		free(header.pixel_types);
		free(header.requested_pixel_types);

		std::cout << "Pixels written to: " << path << std::endl;
	}



	void WriteTextureToBMP(GLTexture2D* texture, const char* path) {
		std::vector<glm::vec3> pixels;
		texture->DownloadTexture(pixels);

		glm::ivec2 size{ texture->GetWidth(), texture->GetHeight()};

		WriteBMP(pixels, size, path);
	}

	void WriteTextureToEXR(GLTexture2D* texture, const char* path) {
		std::vector<glm::vec3> pixels;
		texture->DownloadTexture(pixels);

		glm::ivec2 size{ texture->GetWidth(), texture->GetHeight() };

		WriteEXR(pixels, size, path);
	}
}