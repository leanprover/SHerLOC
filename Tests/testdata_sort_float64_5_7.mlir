"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x7xf64>, res_attrs = [{jax.result_info = "[0]", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<5x7xf64>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<5x7xf64>
    %4 = "stablehlo.sort"(%2) <{dimension = 0 : i64}> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      %5 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
      %6 = "stablehlo.compare"(%arg0, %5) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %7 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
      %8 = "stablehlo.select"(%6, %7, %arg0) : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
      %9 = "stablehlo.compare"(%arg0, %arg0) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %10 = "stablehlo.constant"() <{value = dense<0x7FF8000000000000> : tensor<f64>}> : () -> tensor<f64>
      %11 = "stablehlo.select"(%9, %10, %8) : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
      %12 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
      %13 = "stablehlo.compare"(%arg1, %12) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %14 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
      %15 = "stablehlo.select"(%13, %14, %arg1) : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
      %16 = "stablehlo.compare"(%arg1, %arg1) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %17 = "stablehlo.constant"() <{value = dense<0x7FF8000000000000> : tensor<f64>}> : () -> tensor<f64>
      %18 = "stablehlo.select"(%16, %17, %15) : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
      %19 = "stablehlo.compare"(%11, %18) <{compare_type = #stablehlo<comparison_type TOTALORDER>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<f64>, tensor<f64>) -> tensor<i1>
      "stablehlo.return"(%19) : (tensor<i1>) -> ()
    }) : (tensor<5x7xf64>) -> tensor<5x7xf64>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x7xf64>, tensor<5x7xf64>) -> ()
    "func.return"(%4) : (tensor<5x7xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x7xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-4.0896635946524977, 2.7632953409939716, -0.50213168480988946, -2.0543716182315079, 4.7389017179183535, -5.0992561837764008, 3.2311120719177095], [-3.6800679015698829, -5.2035365268290956, -0.37137121268702389, 3.9131576334488489, 8.2306975220664498, 1.2187365390560596, 2.5896494222636242], [-0.92573122376613293, -0.71528440028190499, 2.6681087215860151, 0.97069439024483861, -2.1846940329152411, -2.2067052115096883, 0.92144406147571478], [-2.8321294558434205, 1.1302784794698757, 3.7176209758875078, -1.3679934880325813, -0.11983810579204865, -5.5580946032460581, -1.8742713896235315], [1.6630315746085409, 1.4959596210409398, -2.0998486044131641, 1.544143027486987, 0.033663053680914386, -1.625141303366769, -0.73063142936550929]]> : tensor<5x7xf64>}> : () -> tensor<5x7xf64>
    "func.return"(%1) : (tensor<5x7xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x7xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-4.0896635946524977, -5.2035365268290956, -2.0998486044131641, -2.0543716182315079, -2.1846940329152411, -5.5580946032460581, -1.8742713896235315], [-3.6800679015698829, -0.71528440028190499, -0.50213168480988946, -1.3679934880325813, -0.11983810579204865, -5.0992561837764008, -0.73063142936550929], [-2.8321294558434205, 1.1302784794698757, -0.37137121268702389, 0.97069439024483861, 0.033663053680914386, -2.2067052115096883, 0.92144406147571478], [-0.92573122376613293, 1.4959596210409398, 2.6681087215860151, 1.544143027486987, 4.7389017179183535, -1.625141303366769, 2.5896494222636242], [1.6630315746085409, 2.7632953409939716, 3.7176209758875078, 3.9131576334488489, 8.2306975220664498, 1.2187365390560596, 3.2311120719177095]]> : tensor<5x7xf64>}> : () -> tensor<5x7xf64>
    "func.return"(%0) : (tensor<5x7xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

