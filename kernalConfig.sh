for B in 2 4 8 16 32 64; do
    for T in 2 4 8 16 32 64; do
        ./taskgpu 60000 "$((B))" "$((T))"
    done
done

