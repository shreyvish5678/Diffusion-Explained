#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dirent.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

double sample_norm() {
    double pi = acos(-1.0);
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2);
    return z0;
}

double* create_schedule(double min, double max, int timesteps) {
    double *schedule = (double *)malloc(timesteps * sizeof(double));
    double val = min;
    for (int t = 0; t < timesteps; t++) {
        schedule[t] = val;
        val += (max - min) / (timesteps - 1);
    }
    return schedule;
}

double* create_alpha_cumprod_schedule(double* beta_schedule, int max_timesteps) {
    double *alpha_cumprod = (double *)malloc(max_timesteps * sizeof(double));
    double alpha_cumprod_t = 1 - beta_schedule[0];
    double alpha_t = 1 - beta_schedule[0];
    for (int t = 0; t < max_timesteps; t++) {
        alpha_cumprod[t] = alpha_cumprod_t;
        if (t == max_timesteps - 1) break;
        alpha_t = 1 - beta_schedule[t + 1];
        alpha_cumprod_t *= alpha_t;
    }
    return alpha_cumprod;
}

double* create_noise(const int* dims) {
    int dim1 = dims[0]; int dim2 = dims[1]; int dim3 = dims[2]; int dim4 = dims[3];
    double* noise = (double*)malloc(dim1 * dim2 * dim3 * dim4 * sizeof(double));
    for (int i = 0; i < dim1 * dim2 * dim3 * dim4; i++) {
        noise[i] = sample_norm();
    }
    return noise;
}

double* forward_diffusion(double* x_0, int* dims, int* t, double* alpha_cumprod) {
    double* noise = create_noise(dims);
    double* x_t = (double*)malloc(dims[0] * dims[1] * dims[2] * dims[3] * sizeof(double));
    double* alpha_t = (double*)malloc(dims[0] * sizeof(double));
    for (int i = 0; i < dims[0]; i++) {
        int t_i = t[i];
        alpha_t[i] = sqrt(alpha_cumprod[t_i - 1]);
    }
    double* beta_t = (double*)malloc(dims[0] * sizeof(double));
    for (int i = 0; i < dims[0]; i++) {
        int t_i = t[i];
        beta_t[i] = sqrt(1 - alpha_cumprod[t_i - 1]);
    }
    for (int i = 0; i < dims[0]; i++) {
        for (int j = 0; j < dims[1] * dims[2] * dims[3]; j++) {
            int idx = i * dims[1] * dims[2] * dims[3] + j;
            x_t[idx] = alpha_t[i] * x_0[idx] + beta_t[i] * noise[idx];
        }
    }
    free(noise);
    free(alpha_t);
    free(beta_t);
    return x_t; 
}

double* model (double* x_t, int* dims, int* t) {
    // Placeholder for the model function
    // In a real implementation, this would be a neural network model
    return NULL;
}

double* sample_ddim(int* dims, int max_timesteps, int sample_timesteps, double* alpha_cumprod) {
    int* ddim_timesteps = (int*)malloc(sample_timesteps * sizeof(int));
    double* ddim_timesteps_double = create_schedule(max_timesteps - 1, 0, sample_timesteps);
    for (int i = 0; i < sample_timesteps; i++) {
        ddim_timesteps[i] = (int)(ddim_timesteps_double[i]);
    }
    double* x_t = create_noise(dims);
    double* x_0_hat = (double*)malloc(dims[0] * dims[1] * dims[2] * dims[3] * sizeof(double));
    double* alpha_t = (double*)malloc(dims[0] * sizeof(double));
    double* alpha_t_next = (double*)malloc(dims[0] * sizeof(double));
    double* beta_t = (double*)malloc(dims[0] * sizeof(double));
    double* beta_t_next = (double*)malloc(dims[0] * sizeof(double));
    int* t_curr = (int*)malloc(dims[0] * sizeof(int));
    int* t_next = (int*)malloc(dims[0] * sizeof(int));
    for (int i = 0; i < sample_timesteps - 1; i++) {
        for (int j = 0; j < dims[0]; j++) {
            t_curr[j] = ddim_timesteps[i];
            t_next[j] = ddim_timesteps[i + 1];
            alpha_t[j] = sqrt(alpha_cumprod[t_curr[j]]);
            beta_t[j] = sqrt(1 - alpha_cumprod[t_curr[j]]);
            alpha_t_next[j] = sqrt(alpha_cumprod[t_next[j]]);
            beta_t_next[j] = sqrt(1 - alpha_cumprod[t_next[j]]);        
        }

        double* pred_noise = model(x_t, dims, t_curr);
        for (int j = 0; j < dims[0]; j++) {
            for (int k = 0; k < dims[1] * dims[2] * dims[3]; k++) {
                int idx = j * dims[1] * dims[2] * dims[3] + k;
                x_0_hat[idx] = (x_t[idx] - beta_t[j] * pred_noise[idx]) / alpha_t[j];
                x_t[idx] = alpha_t_next[j] * x_0_hat[idx] + beta_t_next[j] * pred_noise[idx];
            }
        }
        for (int j = 0; j < dims[0] * dims[1] * dims[2] * dims[3]; j++) {
            if (x_t[j] < -1.0) x_t[j] = -1.0;
            if (x_t[j] > 1.0) x_t[j] = 1.0;
        }
    }
    for (int j = 0; j < dims[0] * dims[1] * dims[2] * dims[3]; j++) {
        x_t[j] = (x_t[j] + 1.0) / 2.0;
    }
    free(ddim_timesteps_double);
    free(ddim_timesteps);
    free(x_0_hat);
    free(alpha_t);
    free(alpha_t_next);
    free(beta_t);
    free(beta_t_next);
    free(t_curr);
    free(t_next);
    return x_t;
}

int main() {
    const char* input_dir = "images/";
    const char* output_dir = "output/";
    int timesteps[] = {50, 100, 250, 500, 1000};

    double* beta_schedule = create_schedule(0.0001, 0.02, 1000);
    double* alpha_cumprod = create_alpha_cumprod_schedule(beta_schedule, 1000);
    free(beta_schedule);

    DIR* dir = opendir(input_dir);
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        if (strstr(entry->d_name, ".png")) {
            char input_path[512];
            snprintf(input_path, sizeof(input_path), "%s%s", input_dir, entry->d_name);

            int width, height, channels;
            unsigned char* image = stbi_load(input_path, &width, &height, &channels, 3); 
            if (!image) {
                fprintf(stderr, "Failed to load %s\n", input_path);
                continue;
            }

            int total_pixels = width * height * 3;
            double* image_float = (double*)malloc(total_pixels * sizeof(double));
            for (int i = 0; i < total_pixels; i++)
                image_float[i] = (image[i] / 255.0) * 2.0 - 1.0;

            int dims[4] = {1, 3, height, width};

            for (int i = 0; i < 3; i++) {
                int t_val = timesteps[i];
                int t_arr[1] = {t_val};

                double* x_t = forward_diffusion(image_float, dims, t_arr, alpha_cumprod);

                unsigned char* out_img = (unsigned char*)malloc(total_pixels);
                for (int j = 0; j < total_pixels; j++) {
                    double val = (x_t[j] + 1.0) / 2.0;
                    if (val < 0) val = 0;
                    if (val > 1) val = 1;
                    out_img[j] = (unsigned char)(val * 255.0);
                }

                char out_path[512];
                snprintf(out_path, sizeof(out_path), "%s%s_t%d.png", output_dir, entry->d_name, t_val);
                stbi_write_png(out_path, width, height, 3, out_img, width * 3);

                free(out_img);
                free(x_t);
            }

            stbi_image_free(image);
            free(image_float);
        }
    }

    closedir(dir);
    free(alpha_cumprod);
    return 0;
}