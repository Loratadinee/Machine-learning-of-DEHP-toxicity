library(gbm)
library(pROC)
library(ggplot2)
library(dplyr)
library(parallel)
library(randomForestSRC)
library(xgboost)
library(Matrix)

train_exp <- read.table("D:/", 
                        header = TRUE, row.names = 1)
train_labels <- read.table("D:/", 
                           header = FALSE)$V2
hup_genes <- read.table("D:/", 
                        header = FALSE)$V1

ml_data <- data.frame(t(train_exp[hup_genes, ]))
ml_data$Class <- as.numeric(ifelse(train_labels == "0", 0, 1))

available_cores <- max(1, parallel::detectCores(logical = FALSE) - 1)
message("Using ", available_cores, " CPU cores")

set.seed(123)
rf_model <- rfsrc(Class ~ .,
                  data = ml_data,
                  ntree = 500,
                  importance = TRUE,
                  nsplit = 10)

imp <- rf_model$importance
imp_threshold <- quantile(imp, probs = 0.5)
selected_features <- names(imp[imp > imp_threshold])
message("Selected ", length(selected_features), " features by RF")

rf_selected_data <- ml_data[, c(selected_features, "Class")]

rf_imp <- data.frame(
  Gene = names(rf_model$importance),
  Importance = as.numeric(rf_model$importance),
  Selected = ifelse(names(rf_model$importance) %in% selected_features, "Yes", "No")
) %>% arrange(desc(Importance))

write.table(rf_imp,
            "C:/",
            sep = "\t",
            quote = FALSE,
            row.names = FALSE)

x_data <- as.matrix(rf_selected_data[, selected_features])
x_label <- rf_selected_data$Class
dtrain <- xgb.DMatrix(data = x_data, label = x_label)

set.seed(123)
xgb_params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "logloss",
  eta = 0.01,
  max_depth = 3,
  subsample = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 5
)

xgb_cv <- xgb.cv(
  params = xgb_params,
  data = dtrain,
  nrounds = 10000,
  nfold = 10,
  verbose = 1,
  early_stopping_rounds = 50,
  maximize = FALSE,
  nthread = available_cores
)

best_iter <- xgb_cv$best_iteration
message("Optimal trees: ", best_iter)

final_xgb <- xgb.train(
  params = xgb_params,
  data = dtrain,
  nrounds = best_iter,
  nthread = available_cores,
  verbose = 1
)

importance_matrix <- xgb.importance(model = final_xgb)
write.table(importance_matrix,
            "C:/",
            sep = "\t",
            quote = FALSE,
            row.names = FALSE)
rf_imp_read <- read.table("C:/",sep = "\t", header = TRUE)
rf_imp_plot <- ggplot(rf_imp_read, aes(x = reorder(Gene, Importance), y = Importance, fill = Selected)) +
  geom_col(width = 0.8) +
  coord_flip(clip = "off") +
  scale_fill_manual(values = c("Yes" = "#ED8D5A", "No" = "#7BC0CD")) +
  labs(title = "RF Feature Importance",
       x = "Gene",
       y = "Importance",
       fill = "Selected") +
  theme_bw() +
  theme(
    panel.grid = element_blank(),
    panel.border = element_blank(),
    axis.line = element_blank(),
    axis.ticks = element_blank(),
    axis.text.y = element_text(size = 15, hjust = 1, margin = margin(r = -28)),
    axis.text.x = element_text(size = 15),
    legend.position = "top",
    plot.margin = margin(5, 10, 5, 5)
  )

ggsave("C:/", 
       plot = rf_imp_plot,
       width = 10, 
       height = max(6, nrow(rf_imp)*0.2))

ggsave("C:/", 
       plot = rf_imp_plot,
       width = 10, 
       height = max(6, nrow(rf_imp)*0.2),
       dpi = 600, compression = "lzw")

cv_log <- as.data.frame(xgb_cv$evaluation_log)
cv_plot <- ggplot(cv_log, aes(x = iter)) +
  geom_line(aes(y = train_logloss_mean, color = "Train")) +
  geom_line(aes(y = test_logloss_mean, color = "CV")) +
  geom_vline(xintercept = best_iter, linetype = "dashed", color = "red") +
  labs(title = "XGBoost CV Logloss",
       x = "Number of Trees",
       y = "Logloss",
       color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        axis.text = element_text(size = 20),
        axis.title = element_text(size = 20),
        plot.title = element_text(size = 25, hjust = 0.5),
        legend.text = element_text(size = 20),
        axis.line = element_line(),
        legend.position = "top") +
  scale_color_manual(values = c("Train" = "#7DC0CD", "CV" = "#ED8D5A"))

ggsave("C:/", 
       plot = cv_plot,
       width = 8, height = 6)

ggsave("C:/", 
       plot = cv_plot,
       width = 8, height = 6,
       dpi = 600, compression = "lzw")


xgb_imp_read <- read.table("C:/", sep = "\t", header = TRUE)
xgb_imp_plot <- ggplot(xgb_imp_read, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_col(aes(fill = Gain), width = 0.8) +
  coord_flip(clip = "off") +
  scale_fill_gradient2(low = "#7bc0cd", mid = "#bfdfd2", high = "#ed8d5a",
                       midpoint = median(xgb_imp_df$Gain)) +
  labs(title = "XGBoost Feature Importance (After RF Selection)",
       x = "", y = "Gain") +
  theme_bw() +
  theme(
    panel.grid = element_blank(),
    panel.border = element_blank(),
    axis.line = element_blank(),
    axis.ticks = element_blank(),
    axis.text.y = element_text(size = 20, hjust = 1, margin = margin(r = -16)),
    axis.text.x = element_text(size = 20),
    plot.title = element_text(size = 18),
    axis.title = element_text(size = 18),
    plot.margin = margin(5, 10, 5, 5)
  )

ggsave("C:/",
       plot = xgb_imp_plot,
       width = 8,
       height = max(6, nrow(xgb_imp_df)*0.5))

ggsave("C:/",
       plot = xgb_imp_plot,
       width = 8,
       height = max(6, nrow(xgb_imp_df)*0.5),
       dpi = 600, compression = "lzw")

prob_train <- predict(final_xgb, dtrain)
auc_train <- auc(roc(x_label, prob_train, quiet = TRUE))
message("Training AUC: ", round(auc_train, 3))

roc_obj <- roc(x_label, prob_train, quiet = TRUE)
roc_df <- data.frame(
  FPR = 1 - roc_obj$specificities,
  TPR = roc_obj$sensitivities
)

roc_plot <- ggplot(roc_df, aes(x = FPR, y = TPR)) +
  geom_line(color = "#E64B35", size = 1) +
  geom_abline(linetype = "dashed", color = "grey") +
  scale_x_continuous(expand = c(0, 0), limits = c(0, 1)) +
  scale_y_continuous(expand = c(0, 0), limits = c(0, 1)) +
  coord_fixed() +
  labs(title = paste("ROC Curve (AUC =", round(auc_train, 3), ")"),
       x = "False Positive Rate",
       y = "True Positive Rate") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        axis.text = element_text(size = 15),
        axis.title = element_text(size = 18),
        plot.title = element_text(size = 18, hjust = 0.5)
  )
