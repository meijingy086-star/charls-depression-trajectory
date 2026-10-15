# 在保留 rf_model 等对象的原 RStudio 会话运行；不重新拟合任何模型。
# 默认导出位置为本网页项目的 artifacts 文件夹。
invisible((function() {
  app_dir <- "C:/Users/29183/Documents/Codex/2026-10-15/z/outputs/trajectory_streamlit"
  required <- c("rf_model", "encoder", "train_completed", "train_set",
                "model_feature_cols", "feature_names")
  absent <- required[!vapply(required, exists, logical(1), inherits = TRUE)]
  if (length(absent)) stop("当前会话缺少：", paste(absent, collapse = "、"))
  pkgs <- c("caret", "randomForest", "jsonlite")
  absent_pkgs <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(absent_pkgs)) stop("请先运行 install.packages(c(",
    paste(sprintf('"%s"', absent_pkgs), collapse = ", "), "))")

  rf <- if (inherits(rf_model, "train")) rf_model$finalModel else rf_model
  if (!inherits(rf, "randomForest") || rf$type != "classification") {
    stop("本导出器适用于当前 randomForest 四分类模型。")
  }
  if (inherits(rf_model, "train") && !is.null(rf_model$preProcess)) {
    stop("RF 存在额外预处理，不能跳过；请先适配导出器。")
  }
  if (any(rf$forest$ncat != 1L)) stop("要求 RF 输入全部为数值编码特征。")
  classes <- as.character(rf$classes)
  if (!setequal(classes, paste0("Class", 1:4))) stop("类别须为 Class1 到 Class4。")
  rf_features <- rownames(rf$importance)
  if (is.null(rf_features)) rf_features <- names(rf$forest$ncat)
  if (length(rf_features) != length(feature_names) ||
      !setequal(rf_features, feature_names)) stop("模型特征与 feature_names 不一致。")
  encoded_cols <- model_feature_cols[match(rf_features, feature_names)]
  predictors <- setdiff(names(train_completed), "trajectory_group")
  origin_of <- function(col) {
    hits <- predictors[col == make.names(predictors) |
      startsWith(col, paste0(make.names(predictors), "."))]
    if (!length(hits)) stop("无法映射编码列：", col)
    hits[which.max(nchar(make.names(hits)))]
  }
  origins <- vapply(encoded_cols, origin_of, character(1))
  selected <- unique(unname(origins))
  continuous <- c("age", "sleep", "systo", "diasto", "bmi")
  labels <- c(gender="性别", marry="婚姻状况", edu="受教育程度", retire="退休状态",
    systo="收缩压", diasto="舒张压", bmi="BMI", drinkl="饮酒", social="社会参与",
    chronic="慢性疾病", hibpe="高血压", diabe="糖尿病", arther="关节炎",
    disability="残疾状态", hear="听力", eyesight_close="近距离视力",
    eyesight_distance="远距离视力", sleep="睡眠时间", satlife="生活满意度", age="年龄")
  maps <- list(gender=c("1"="女","2"="男"), marry=c("1"="其他","2"="已婚"),
    edu=c("1"="小学以下","2"="小学","3"="初中","4"="高中及以上"),
    retire=c("1"="未退休","2"="已退休"), drinkl=c("1"="无","2"="有"),
    social=c("1"="无","2"="有"), chronic=c("1"="无","2"="有"),
    hibpe=c("1"="无","2"="有"), diabe=c("1"="无","2"="有"),
    arther=c("1"="无","2"="有"), disability=c("1"="无","2"="有"),
    hear=c("1"="良好","2"="一般","3"="较差"),
    eyesight_close=c("1"="良好","2"="一般","3"="较差"),
    eyesight_distance=c("1"="良好","2"="一般","3"="较差"),
    satlife=c("1"="较差","2"="一般","3"="良好"))
  units <- c(age="岁", sleep="小时/晚", systo="mmHg", diasto="mmHg", bmi="kg/m²")
  if (any(!selected %in% names(labels))) stop("存在未配置的原始变量。")

  # 参考资料由开发集的边际中位数/众数组成，不导出任何真实受试者记录。
  reference <- train_completed[1, , drop = FALSE]
  for (nm in predictors) {
    x <- train_completed[[nm]]
    if (is.factor(x)) {
      counts <- table(x)
      reference[[nm]] <- factor(names(counts)[which.max(counts)], levels=levels(x), ordered=is.ordered(x))
    } else if (is.numeric(x)) reference[[nm]] <- stats::median(x, na.rm=TRUE)
    else stop("未支持的预测变量类型：", nm)
  }
  encode <- function(d) {
    x <- as.matrix(predict(encoder, newdata=d))
    if (!all(encoded_cols %in% colnames(x))) stop("编码器输出缺少模型特征。")
    x <- x[, encoded_cols, drop=FALSE]
    colnames(x) <- rf_features
    if (any(!is.finite(x))) stop("编码后有非有限数值。")
    x
  }
  ref_x <- encode(reference)
  input_specs <- lapply(selected, function(nm) {
    spec <- list(name=nm, label=unname(labels[[nm]]))
    if (nm %in% continuous) {
      x <- train_completed[[nm]]
      spec$type <- "number"
      spec$unit <- unname(units[[nm]])
      spec$default <- as.numeric(reference[[nm]])
      spec$observed_min <- min(x, na.rm=TRUE)
      spec$observed_max <- max(x, na.rm=TRUE)
    } else {
      lev <- levels(train_completed[[nm]])
      mapping <- maps[[nm]]
      if (is.null(mapping) || any(!lev %in% names(mapping))) stop(nm, " 的编码与赋值表不一致。")
      spec$type <- "category"
      spec$default <- as.character(reference[[nm]])
      spec$options <- lapply(lev, function(code) list(value=code, label=unname(mapping[[code]])))
    }
    spec
  })
  feature_specs <- lapply(seq_along(rf_features), function(j) {
    nm <- origins[[j]]
    spec <- list(name=rf_features[[j]], original=nm)
    if (nm %in% continuous) {
      trial <- reference
      trial[[nm]] <- reference[[nm]] + 1
      if (abs(ref_x[1,j] - as.numeric(reference[[nm]])) > 1e-10 ||
          abs(encode(trial)[1,j] - as.numeric(trial[[nm]])) > 1e-10) {
        stop("连续特征并非恒等编码：", rf_features[[j]])
      }
      spec$type <- "number"
    } else {
      lev <- levels(train_completed[[nm]])
      values <- vapply(lev, function(code) {
        trial <- reference
        trial[[nm]] <- factor(code, levels=lev)
        as.numeric(encode(trial)[1,j])
      }, numeric(1))
      spec$type <- "category"
      spec$values <- as.list(setNames(values, lev))
    }
    spec
  })
  portable_encode <- function(d) {
    x <- matrix(NA_real_, nrow=nrow(d), ncol=length(feature_specs), dimnames=list(NULL, rf_features))
    for (j in seq_along(feature_specs)) {
      spec <- feature_specs[[j]]
      if (spec$type == "number") x[,j] <- as.numeric(d[[spec$original]])
      else x[,j] <- unlist(spec$values[as.character(d[[spec$original]])], use.names=FALSE)
    }
    if (any(!is.finite(x))) stop("便携编码产生缺失或无穷值。")
    x
  }
  # 对整个已完成插补的开发集核对转换，避免仅核对一条参考资料。
  expected_x <- encode(train_completed)
  mapped_x <- portable_encode(train_completed)
  if (max(abs(expected_x - mapped_x)) > 1e-10) stop("便携编码与原编码器不一致。")
  if (nrow(train_set) != nrow(expected_x) ||
      max(abs(as.matrix(train_set[,rf_features,drop=FALSE]) - expected_x)) > 1e-10) {
    stop("train_set 与编码后的原始开发集不一致，不能导出。")
  }

  trees <- lapply(seq_len(rf$ntree), function(i) unname(randomForest::getTree(rf, k=i, labelVar=FALSE)))
  portable_prob <- function(x) {
    votes <- matrix(0, nrow(x), length(classes), dimnames=list(NULL, classes))
    for (tree in trees) {
      for (i in seq_len(nrow(x))) {
        node <- 1L
        while (tree[node,5] != -1L) {
          node <- as.integer(if (x[i,tree[node,3]] <= tree[node,4]) tree[node,1] else tree[node,2])
          if (node < 1L || node > nrow(tree)) stop("树节点索引无效。")
        }
        k <- as.integer(tree[node,6])
        votes[i,k] <- votes[i,k] + 1
      }
    }
    votes / length(trees)
  }
  # 人工构造的核对样本覆盖每一分类水平及连续变量的范围端点。
  cases <- list(reference)
  for (spec in input_specs) {
    nm <- spec$name
    values <- if (spec$type == "number") unique(c(spec$observed_min,spec$default,spec$observed_max)) else
      vapply(spec$options, function(z) z$value, character(1))
    for (value in values) {
      trial <- reference
      if (spec$type == "number") trial[[nm]] <- as.numeric(value)
      else trial[[nm]] <- factor(as.character(value), levels=levels(reference[[nm]]))
      cases[[length(cases)+1L]] <- trial
    }
  }
  case_data <- do.call(rbind, cases)
  real_prob <- as.matrix(predict(rf_model, newdata=as.data.frame(encode(case_data)), type="prob"))[,classes,drop=FALSE]
  exported_prob <- portable_prob(encode(case_data))
  max_error <- max(abs(real_prob - exported_prob))
  if (!is.finite(max_error) || max_error > 1e-12) stop("树预测不一致，最大误差：", max_error)
  # 树阈值两侧也作测试；避免分裂点的 < / <= 实现不一致。
  boundary_x <- ref_x[rep(1L, 1L), , drop=FALSE]
  for (tree in trees[seq_len(min(10L,length(trees)))]) {
    nodes <- head(which(tree[,5] != -1L), 3L)
    for (node in nodes) for (delta in c(-1e-8,0,1e-8)) {
      z <- ref_x; z[1,tree[node,3]] <- tree[node,4] + delta
      boundary_x <- rbind(boundary_x,z)
    }
  }
  boundary_error <- max(abs(portable_prob(boundary_x) -
    as.matrix(predict(rf_model,newdata=as.data.frame(boundary_x),type="prob"))[,classes,drop=FALSE]))
  if (!is.finite(boundary_error) || boundary_error > 1e-12) stop("树分裂边界核对失败。")
  verification_cases <- lapply(seq_len(nrow(case_data)), function(i) {
    raw_input <- lapply(selected, function(nm) {
      if (nm %in% continuous) as.numeric(case_data[[nm]][i]) else as.character(case_data[[nm]][i])
    })
    names(raw_input) <- selected
    list(inputs=raw_input, encoded=as.list(unname(encode(case_data[i,,drop=FALSE])[1,])),
         probabilities=as.list(unname(real_prob[i,])))
  })

  metrics <- list()
  if (exists("metrics_summary_fixed", inherits=TRUE)) {
    mm <- metrics_summary_fixed
    metrics <- lapply(seq_len(nrow(mm)), function(i) as.list(mm[i,,drop=FALSE]))
  }
  trajectories <- list()
  if (exists("df_long_dev", inherits=TRUE) && exists("m4_lcgm", inherits=TRUE)) {
    dd <- as.data.frame(df_long_dev)
    pp <- m4_lcgm$pprob
    dd$class <- pp$class[match(dd$id_num,pp$id_num)]
    dd <- dd[!is.na(dd$class) & !is.na(dd$cesd10),,drop=FALSE]
    means <- stats::aggregate(cesd10 ~ class + wave, data=dd, FUN=mean)
    years <- c("1"=2011,"2"=2013,"3"=2015,"4"=2018,"5"=2020)
    trajectories <- lapply(seq_len(nrow(means)), function(i) list(
      class=paste0("Class",means$class[i]), year=unname(years[as.character(means$wave[i])]),
      mean=means$cesd10[i]))
  }
  model <- list(format="charls-rf-v1", model_type="randomForest classification",
    created_at=format(Sys.time(),tz="UTC",usetz=TRUE),
    classes=as.list(classes), inputs=input_specs, features=feature_specs,
    trees=trees, n_trees=length(trees), tie_method="first",
    development_n=nrow(train_set),
    validation_n=if (exists("test_set",inherits=TRUE)) nrow(test_set) else NULL,
    development_class_counts=as.list(table(train_set$trajectory_group)),
    validation_class_counts=if (exists("test_set",inherits=TRUE)) as.list(table(test_set$trajectory_group)) else NULL,
    metrics=metrics, trajectories=trajectories,
    verification_cases=verification_cases,
    parity=list(case_count=nrow(case_data), max_probability_error=max_error,
                boundary_case_count=nrow(boundary_x), boundary_probability_error=boundary_error,
                full_development_encoding_rows=nrow(mapped_x)),
    notes=list(probability="未校准的随机森林投票比例；最终拟合使用开发集上采样。",
      missing="网页要求所有入选变量完整输入，不在新受试者上重新运行MICE。",
      validation="随机划分内部验证；模型结果仅供研究。"))
  dest <- file.path(app_dir,"artifacts")
  dir.create(dest,recursive=TRUE,showWarnings=FALSE)
  path <- file.path(dest,"model.json.gz")
  con <- gzfile(path,open="wb")
  tryCatch(jsonlite::write_json(model,con,auto_unbox=TRUE,digits=NA,na="null",
                               null="null",matrix="rowmajor",pretty=FALSE),finally=close(con))
  # 用R重新读取导出结果，检查最关键字段未在序列化时改变结构。
  con <- gzfile(path,open="rt",encoding="UTF-8")
  reloaded <- tryCatch(jsonlite::fromJSON(con,simplifyVector=FALSE),finally=close(con))
  if (length(reloaded$trees) != rf$ntree || length(reloaded$classes) != 4L) stop("导出结构核对失败。")
  cat("\n真实随机森林模型已导出：\n",normalizePath(path,winslash="/"),"\n",
      "原始输入变量数：",length(input_specs),"；编码特征数：",length(feature_specs),
      "；树数：",rf$ntree,"\n",
      "人工合成样本及阈值边界的最大预测误差：",max(max_error,boundary_error),"\n",
      "未重新训练；未导出受试者ID或逐人记录。\n",sep="")
})())
