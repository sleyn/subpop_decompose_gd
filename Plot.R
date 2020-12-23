library(here)
setwd(here())

library(tidyverse)
library(viridis)
library(ggrepel)

tests = read_tsv('Test/Test/Test_results.tsv')
tests = tests %>% mutate(Size = Variants * Samples)

tests_long = tests %>% 
  filter(MSE_SQRT <= 1) %>%
  select(Size, Exec_Time, F1_score, MSE_SQRT, Precision, Recall) %>%
  pivot_longer(-Size, names_to='Test', values_to='Score')

plot_1 = tests_long %>%
  ggplot(aes(Size, Score)) +
  geom_boxplot(aes(group = cut_width(Size, 25)), outlier.alpha = 0) +
  geom_smooth(color = "blue") +
  geom_point(alpha = 0.2) +
  facet_grid(Test ~ ., scales = "free_y") +
  theme_bw()

ggsave('Size_effect.png', height = 8, width = 6)

plot_2 = tests %>% ggplot(aes(Variants, Samples, color=F1_score, label=F1_score)) + 
  geom_point(size = 4, alpha = 0.5) +
  geom_text_repel(size = 2) +
  scale_color_viridis(name = "F1 score") +
  scale_y_continuous(breaks = 1:10, minor_breaks = F) +
  xlab('Number of variants') +
  ylab('Number of samples') +
  theme_bw()

ggsave('F1_Size.png', height = 6, width = 6)

plot_3 = tests %>% filter(MSE_SQRT <= 1) %>% ggplot(aes(F1_score, MSE_SQRT)) +
  geom_density2d_filled() +
  geom_point(shape = 1, color = "white") +
  theme_minimal() +
  xlab('F1 Score') +
  ylab('Square root of MSE')

ggsave('F1_MSE.png', height = 6, width = 6)
