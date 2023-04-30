#include <stdint.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define add_ 0x269ec3ULL
#define mul_ 0x5d588b656c078965ULL
#define BLOCKSIZE 128


__device__ __inline__ uint32_t change_endian(uint32_t val) {
	val = ((val << 8) & 0xFF00FF00U) | ((val >> 8) & 0xFF00FFU);
	return (val << 16) | (val >> 16);
}

__device__ __forceinline__ int sidx(int thread_id, int logical_index)
{
	return logical_index * BLOCKSIZE + thread_id;
}

__global__ void kernel_vending_search(uint64_t* seeds, uint64_t const mask, uint32_t const size, int32_t const range) {
	const uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
	if (size <= i)return;

	uint64_t state = seeds[i];
	uint64_t checkcode = 0xFFFFFFFFFFFFFFFFULL;
	bool result = 0UL;

	for (uint32_t j = 0; j < range; ++j) {
		checkcode <<= 5;
		checkcode ^= (state>>32) >> 27;
		result |= ((checkcode & mask) == 0ULL);
		state *= mul_;
		state += add_;
	}
	seeds[i] = result * (i+1);
}

__global__ void kernel_generate_initseed(uint32_t* params, uint64_t* initseeds, uint64_t* seeds, uint32_t const size, uint64_t const initmul, uint64_t const initadd, 
	uint32_t const m0, uint32_t const m1, uint32_t const m2, uint32_t const m3, uint32_t const m4, uint32_t const m5, uint32_t const m6, uint32_t const m7) {
	const uint32_t pos = blockDim.x * blockIdx.x + threadIdx.x;
	if (size <= pos)return;
	__shared__ uint32_t W[80 * BLOCKSIZE];

	W[sidx(threadIdx.x, 0)] = change_endian(m0);
	W[sidx(threadIdx.x, 1)] = change_endian(m1);
	W[sidx(threadIdx.x, 2)] = change_endian(m2);
	W[sidx(threadIdx.x, 3)] = change_endian(m3);
	W[sidx(threadIdx.x, 4)] = change_endian(m4);
	W[sidx(threadIdx.x, 5)] = change_endian(m5 ^ params[4 * pos]);
	W[sidx(threadIdx.x, 6)] = change_endian(m6);
	W[sidx(threadIdx.x, 7)] = change_endian(m7);
	W[sidx(threadIdx.x, 8)] = change_endian(params[4 * pos + 1]);
	W[sidx(threadIdx.x, 9)] = change_endian(params[4 * pos + 2]);
	W[sidx(threadIdx.x, 10)] = 0x00000000U;
	W[sidx(threadIdx.x, 11)] = 0x00000000U;
	W[sidx(threadIdx.x, 12)] = change_endian(params[4 * pos + 3]);


	W[sidx(threadIdx.x, 13)] = 0x80000000U;
	W[sidx(threadIdx.x, 14)] = 0x00000000U;
	W[sidx(threadIdx.x, 15)] = 0x000001A0U;
	uint32_t t;
	for (t = 16; t < 80; t++)
	{
		uint32_t w = W[sidx(threadIdx.x, t - 3)] ^ W[sidx(threadIdx.x, t - 8)] ^ W[sidx(threadIdx.x, t - 14)] ^ W[sidx(threadIdx.x, t - 16)];
		W[sidx(threadIdx.x, t)] = (w << 1) | (w >> 31);
	}
	const uint32_t H0 = 0x67452301;
	const uint32_t H1 = 0xEFCDAB89;
	const uint32_t H2 = 0x98BADCFE;
	const uint32_t H3 = 0x10325476;
	const uint32_t H4 = 0xC3D2E1F0;

	uint32_t A, B, C, D, E;
	A = H0; B = H1; C = H2; D = H3; E = H4;

	for (t = 0; t < 20; t++)
	{
		uint32_t temp = ((A << 5) | (A >> 27)) + ((B & C) | ((~B) & D)) + E + W[sidx(threadIdx.x, t)] + 0x5A827999;
		E = D;
		D = C;
		C = (B << 30) | (B >> 2);
		B = A;
		A = temp;
	}
	for (; t < 40; t++)
	{
		uint32_t temp = ((A << 5) | (A >> 27)) + (B ^ C ^ D) + E + W[sidx(threadIdx.x, t)] + 0x6ED9EBA1;
		E = D;
		D = C;
		C = (B << 30) | (B >> 2);
		B = A;
		A = temp;
	}
	for (; t < 60; t++)
	{
		uint32_t temp = ((A << 5) | (A >> 27)) + ((B & C) | (B & D) | (C & D)) + E + W[sidx(threadIdx.x, t)] + 0x8F1BBCDC;
		E = D;
		D = C;
		C = (B << 30) | (B >> 2);
		B = A;
		A = temp;
	}
	for (; t < 80; t++)
	{
		uint32_t temp = ((A << 5) | (A >> 27)) + (B ^ C ^ D) + E + W[sidx(threadIdx.x, t)] + 0xCA62C1D6;
		E = D;
		D = C;
		C = (B << 30) | (B >> 2);
		B = A;
		A = temp;
	}

	uint64_t initseed = change_endian(H1 + B);
	initseed <<= 32;
	initseed |= change_endian(H0 + A);
	initseed = initseed * mul_ + add_;
	initseeds[pos] = initseed;
	seeds[pos] = initseed * initmul + initadd;
}

__host__ void device_vending_search(uint64_t* seeds, uint64_t const mask, uint32_t const size, int32_t const range) {
	uint32_t block = BLOCKSIZE;
	uint32_t grid = (size + block - 1) / block;
	kernel_vending_search<<<grid, block>>>(seeds, mask, size, range);
}

__host__ void device_generate_initseed(uint32_t* params, uint64_t* initseeds, uint64_t* seeds, uint32_t const size, uint64_t const initmul, uint64_t const initadd, uint32_t const m0, uint32_t const m1, uint32_t const m2, uint32_t const m3, uint32_t const m4, uint32_t const m5, uint32_t const m6, uint32_t const m7) {
	uint32_t block = BLOCKSIZE;
	uint32_t grid = (size + block - 1) / block;
	kernel_generate_initseed << <grid, block >> > (params, initseeds, seeds, size, initmul, initadd, m0, m1, m2, m3, m4, m5, m6, m7);
}