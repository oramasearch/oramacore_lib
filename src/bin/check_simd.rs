use oramacore_lib::data_structures::hnsw::IS_SIMD_SUPPORTED;

fn main() {
    let is_supported = &*IS_SIMD_SUPPORTED;
    println!("IS_SIMD_SUPPORTED: {is_supported}");
}
