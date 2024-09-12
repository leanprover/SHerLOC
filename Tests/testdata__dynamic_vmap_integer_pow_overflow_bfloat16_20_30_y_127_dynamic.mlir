"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x20x30xbf16>) -> tensor<?x20x30xbf16>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg2: tensor<i64>, %arg3: tensor<?x20x30xbf16>):
    %12 = "func.call"(%arg2, %arg3) <{callee = @integer_pow}> : (tensor<i64>, tensor<?x20x30xbf16>) -> tensor<?x20x30xbf16>
    "func.return"(%12) : (tensor<?x20x30xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<?x20x30xbf16>) -> tensor<?x20x30xbf16>, sym_name = "integer_pow", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x20x30xbf16>):
    %0 = "stablehlo.multiply"(%arg1, %arg1) : (tensor<?x20x30xbf16>, tensor<?x20x30xbf16>) -> tensor<?x20x30xbf16>
    %1 = "stablehlo.multiply"(%arg1, %0) : (tensor<?x20x30xbf16>, tensor<?x20x30xbf16>) -> tensor<?x20x30xbf16>
    %2 = "stablehlo.multiply"(%0, %0) : (tensor<?x20x30xbf16>, tensor<?x20x30xbf16>) -> tensor<?x20x30xbf16>
    %3 = "stablehlo.multiply"(%1, %2) : (tensor<?x20x30xbf16>, tensor<?x20x30xbf16>) -> tensor<?x20x30xbf16>
    %4 = "stablehlo.multiply"(%2, %2) : (tensor<?x20x30xbf16>, tensor<?x20x30xbf16>) -> tensor<?x20x30xbf16>
    %5 = "stablehlo.multiply"(%3, %4) : (tensor<?x20x30xbf16>, tensor<?x20x30xbf16>) -> tensor<?x20x30xbf16>
    %6 = "stablehlo.multiply"(%4, %4) : (tensor<?x20x30xbf16>, tensor<?x20x30xbf16>) -> tensor<?x20x30xbf16>
    %7 = "stablehlo.multiply"(%5, %6) : (tensor<?x20x30xbf16>, tensor<?x20x30xbf16>) -> tensor<?x20x30xbf16>
    %8 = "stablehlo.multiply"(%6, %6) : (tensor<?x20x30xbf16>, tensor<?x20x30xbf16>) -> tensor<?x20x30xbf16>
    %9 = "stablehlo.multiply"(%7, %8) : (tensor<?x20x30xbf16>, tensor<?x20x30xbf16>) -> tensor<?x20x30xbf16>
    %10 = "stablehlo.multiply"(%8, %8) : (tensor<?x20x30xbf16>, tensor<?x20x30xbf16>) -> tensor<?x20x30xbf16>
    %11 = "stablehlo.multiply"(%9, %10) : (tensor<?x20x30xbf16>, tensor<?x20x30xbf16>) -> tensor<?x20x30xbf16>
    "func.return"(%11) : (tensor<?x20x30xbf16>) -> ()
  }) : () -> ()
}) : () -> ()

