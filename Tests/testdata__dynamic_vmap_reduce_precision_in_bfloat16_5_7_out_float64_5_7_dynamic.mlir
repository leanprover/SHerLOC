"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x5x7xbf16>) -> tensor<?x5x7xbf16>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x5x7xbf16>):
    %0 = "stablehlo.reduce_precision"(%arg1) <{exponent_bits = 11 : i32, mantissa_bits = 52 : i32}> : (tensor<?x5x7xbf16>) -> tensor<?x5x7xbf16>
    "func.return"(%0) : (tensor<?x5x7xbf16>) -> ()
  }) : () -> ()
}) : () -> ()

