"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x20x20xbf16>) -> tensor<?x20x20xbf16>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x20x20xbf16>):
    %0 = "stablehlo.constant"() <{value = dense<-1.000000e+00> : tensor<bf16>}> : () -> tensor<bf16>
    %1 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %2 = "stablehlo.reshape"(%1) : (tensor<i32>) -> tensor<1xi32>
    %3 = "stablehlo.constant"() <{value = dense<20> : tensor<1xi32>}> : () -> tensor<1xi32>
    %4 = "stablehlo.constant"() <{value = dense<20> : tensor<1xi32>}> : () -> tensor<1xi32>
    %5 = "stablehlo.concatenate"(%2, %3, %4) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %6 = "stablehlo.dynamic_broadcast_in_dim"(%0, %5) <{broadcast_dimensions = array<i64>}> : (tensor<bf16>, tensor<3xi32>) -> tensor<?x20x20xbf16>
    %7 = "stablehlo.compare"(%arg1, %6) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<?x20x20xbf16>, tensor<?x20x20xbf16>) -> tensor<?x20x20xi1>
    %8 = "stablehlo.multiply"(%arg1, %arg1) : (tensor<?x20x20xbf16>, tensor<?x20x20xbf16>) -> tensor<?x20x20xbf16>
    %9 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<bf16>}> : () -> tensor<bf16>
    %10 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %11 = "stablehlo.reshape"(%10) : (tensor<i32>) -> tensor<1xi32>
    %12 = "stablehlo.constant"() <{value = dense<20> : tensor<1xi32>}> : () -> tensor<1xi32>
    %13 = "stablehlo.constant"() <{value = dense<20> : tensor<1xi32>}> : () -> tensor<1xi32>
    %14 = "stablehlo.concatenate"(%11, %12, %13) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %15 = "stablehlo.dynamic_broadcast_in_dim"(%9, %14) <{broadcast_dimensions = array<i64>}> : (tensor<bf16>, tensor<3xi32>) -> tensor<?x20x20xbf16>
    %16 = "stablehlo.subtract"(%15, %8) : (tensor<?x20x20xbf16>, tensor<?x20x20xbf16>) -> tensor<?x20x20xbf16>
    %17 = "stablehlo.sqrt"(%16) : (tensor<?x20x20xbf16>) -> tensor<?x20x20xbf16>
    %18 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<bf16>}> : () -> tensor<bf16>
    %19 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %20 = "stablehlo.reshape"(%19) : (tensor<i32>) -> tensor<1xi32>
    %21 = "stablehlo.constant"() <{value = dense<20> : tensor<1xi32>}> : () -> tensor<1xi32>
    %22 = "stablehlo.constant"() <{value = dense<20> : tensor<1xi32>}> : () -> tensor<1xi32>
    %23 = "stablehlo.concatenate"(%20, %21, %22) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %24 = "stablehlo.dynamic_broadcast_in_dim"(%18, %23) <{broadcast_dimensions = array<i64>}> : (tensor<bf16>, tensor<3xi32>) -> tensor<?x20x20xbf16>
    %25 = "stablehlo.add"(%24, %arg1) : (tensor<?x20x20xbf16>, tensor<?x20x20xbf16>) -> tensor<?x20x20xbf16>
    %26 = "stablehlo.atan2"(%17, %25) : (tensor<?x20x20xbf16>, tensor<?x20x20xbf16>) -> tensor<?x20x20xbf16>
    %27 = "stablehlo.constant"() <{value = dense<2.000000e+00> : tensor<bf16>}> : () -> tensor<bf16>
    %28 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %29 = "stablehlo.reshape"(%28) : (tensor<i32>) -> tensor<1xi32>
    %30 = "stablehlo.constant"() <{value = dense<20> : tensor<1xi32>}> : () -> tensor<1xi32>
    %31 = "stablehlo.constant"() <{value = dense<20> : tensor<1xi32>}> : () -> tensor<1xi32>
    %32 = "stablehlo.concatenate"(%29, %30, %31) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %33 = "stablehlo.dynamic_broadcast_in_dim"(%27, %32) <{broadcast_dimensions = array<i64>}> : (tensor<bf16>, tensor<3xi32>) -> tensor<?x20x20xbf16>
    %34 = "stablehlo.multiply"(%33, %26) : (tensor<?x20x20xbf16>, tensor<?x20x20xbf16>) -> tensor<?x20x20xbf16>
    %35 = "stablehlo.constant"() <{value = dense<3.140630e+00> : tensor<bf16>}> : () -> tensor<bf16>
    %36 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %37 = "stablehlo.reshape"(%36) : (tensor<i32>) -> tensor<1xi32>
    %38 = "stablehlo.constant"() <{value = dense<20> : tensor<1xi32>}> : () -> tensor<1xi32>
    %39 = "stablehlo.constant"() <{value = dense<20> : tensor<1xi32>}> : () -> tensor<1xi32>
    %40 = "stablehlo.concatenate"(%37, %38, %39) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %41 = "stablehlo.dynamic_broadcast_in_dim"(%35, %40) <{broadcast_dimensions = array<i64>}> : (tensor<bf16>, tensor<3xi32>) -> tensor<?x20x20xbf16>
    %42 = "stablehlo.select"(%7, %34, %41) : (tensor<?x20x20xi1>, tensor<?x20x20xbf16>, tensor<?x20x20xbf16>) -> tensor<?x20x20xbf16>
    "func.return"(%42) : (tensor<?x20x20xbf16>) -> ()
  }) : () -> ()
}) : () -> ()

