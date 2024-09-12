"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %5:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<20x20xf32>, tensor<1x20xf32>)
    %6 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xf32>
    %7 = "stablehlo.broadcast_in_dim"(%5#1) <{broadcast_dimensions = array<i64: 0, 1>}> : (tensor<1x20xf32>) -> tensor<20x20xf32>
    %8 = "stablehlo.compare"(%5#0, %5#0) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %9 = "stablehlo.compare"(%7, %7) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %10 = "stablehlo.or"(%8, %9) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %11 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %12 = "stablehlo.broadcast_in_dim"(%11) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %13 = "stablehlo.compare"(%7, %12) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %14 = "stablehlo.constant"() <{value = dense<0x7F800000> : tensor<f32>}> : () -> tensor<f32>
    %15 = "stablehlo.broadcast_in_dim"(%14) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %16 = "stablehlo.compare"(%7, %15) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %17 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %18 = "stablehlo.broadcast_in_dim"(%17) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %19 = "stablehlo.compare"(%7, %18) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %20 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %21 = "stablehlo.broadcast_in_dim"(%20) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %22 = "stablehlo.compare"(%5#0, %21) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %23 = "stablehlo.or"(%19, %22) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %24 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %25 = "stablehlo.broadcast_in_dim"(%24) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %26 = "stablehlo.compare"(%7, %25) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %27 = "stablehlo.compare"(%7, %5#0) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %28 = "stablehlo.and"(%26, %27) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %29 = "stablehlo.log"(%7) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %30 = "stablehlo.multiply"(%5#0, %29) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %31 = "stablehlo.subtract"(%30, %7) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %32 = "chlo.lgamma"(%5#0) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %33 = "stablehlo.subtract"(%31, %32) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %34 = "stablehlo.constant"() <{value = dense<3.40282347E+38> : tensor<f32>}> : () -> tensor<f32>
    %35 = "stablehlo.log"(%34) : (tensor<f32>) -> tensor<f32>
    %36 = "stablehlo.negate"(%35) : (tensor<f32>) -> tensor<f32>
    %37 = "stablehlo.broadcast_in_dim"(%36) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %38 = "stablehlo.compare"(%33, %37) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %39 = "stablehlo.exponential"(%33) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %40 = "stablehlo.or"(%13, %23) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %41 = "stablehlo.or"(%40, %38) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %42 = "stablehlo.or"(%41, %10) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %43 = "stablehlo.not"(%42) : (tensor<20x20xi1>) -> tensor<20x20xi1>
    %44 = "stablehlo.and"(%43, %28) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %45 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %46 = "stablehlo.broadcast_in_dim"(%45) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %47 = "stablehlo.subtract"(%46, %5#0) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %48 = "stablehlo.add"(%7, %47) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %49 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %50 = "stablehlo.broadcast_in_dim"(%49) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %51 = "stablehlo.add"(%48, %50) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %52 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %53 = "stablehlo.broadcast_in_dim"(%52) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %54 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %55 = "stablehlo.broadcast_in_dim"(%54) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %56 = "stablehlo.add"(%7, %55) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %57 = "stablehlo.multiply"(%51, %7) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %58 = "stablehlo.divide"(%56, %57) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %59 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %60 = "stablehlo.broadcast_in_dim"(%59) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %61 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %62 = "stablehlo.broadcast_in_dim"(%61) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %63 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %64 = "stablehlo.broadcast_in_dim"(%63) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %65 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %66 = "stablehlo.broadcast_in_dim"(%65) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %67 = "stablehlo.negate"(%7) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %68 = "stablehlo.multiply"(%58, %67) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %69 = "stablehlo.subtract"(%66, %68) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %70 = "stablehlo.divide"(%69, %57) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %71 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %72:15 = "stablehlo.while"(%44, %58, %60, %47, %51, %71, %56, %57, %53, %7, %62, %64, %66, %67, %70) ({
    ^bb0(%arg32: tensor<20x20xi1>, %arg33: tensor<20x20xf32>, %arg34: tensor<20x20xf32>, %arg35: tensor<20x20xf32>, %arg36: tensor<20x20xf32>, %arg37: tensor<f32>, %arg38: tensor<20x20xf32>, %arg39: tensor<20x20xf32>, %arg40: tensor<20x20xf32>, %arg41: tensor<20x20xf32>, %arg42: tensor<20x20xf32>, %arg43: tensor<20x20xf32>, %arg44: tensor<20x20xf32>, %arg45: tensor<20x20xf32>, %arg46: tensor<20x20xf32>):
      %232 = "stablehlo.constant"() <{value = dense<2.000000e+03> : tensor<f32>}> : () -> tensor<f32>
      %233 = "stablehlo.compare"(%arg37, %232) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %234 = "stablehlo.constant"() <{value = dense<false> : tensor<i1>}> : () -> tensor<i1>
      %235 = "stablehlo.reduce"(%arg32, %234) <{dimensions = array<i64: 0, 1>}> ({
      ^bb0(%arg47: tensor<i1>, %arg48: tensor<i1>):
        %237 = "stablehlo.or"(%arg47, %arg48) : (tensor<i1>, tensor<i1>) -> tensor<i1>
        "stablehlo.return"(%237) : (tensor<i1>) -> ()
      }) : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
      %236 = "stablehlo.and"(%233, %235) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      "stablehlo.return"(%236) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg17: tensor<20x20xi1>, %arg18: tensor<20x20xf32>, %arg19: tensor<20x20xf32>, %arg20: tensor<20x20xf32>, %arg21: tensor<20x20xf32>, %arg22: tensor<f32>, %arg23: tensor<20x20xf32>, %arg24: tensor<20x20xf32>, %arg25: tensor<20x20xf32>, %arg26: tensor<20x20xf32>, %arg27: tensor<20x20xf32>, %arg28: tensor<20x20xf32>, %arg29: tensor<20x20xf32>, %arg30: tensor<20x20xf32>, %arg31: tensor<20x20xf32>):
      %128 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %129 = "stablehlo.add"(%arg22, %128) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      %130 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %131 = "stablehlo.broadcast_in_dim"(%130) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %132 = "stablehlo.add"(%arg20, %131) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %133 = "stablehlo.constant"() <{value = dense<2.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %134 = "stablehlo.broadcast_in_dim"(%133) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %135 = "stablehlo.add"(%arg21, %134) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %136 = "stablehlo.broadcast_in_dim"(%129) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %137 = "stablehlo.multiply"(%132, %136) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %138 = "stablehlo.multiply"(%arg23, %135) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %139 = "stablehlo.multiply"(%arg25, %137) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %140 = "stablehlo.subtract"(%138, %139) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %141 = "stablehlo.multiply"(%arg24, %135) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %142 = "stablehlo.multiply"(%arg26, %137) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %143 = "stablehlo.subtract"(%141, %142) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %144 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %145 = "stablehlo.broadcast_in_dim"(%144) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %146 = "stablehlo.compare"(%143, %145) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %147 = "stablehlo.divide"(%140, %143) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %148 = "stablehlo.subtract"(%arg18, %147) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %149 = "stablehlo.divide"(%148, %147) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %150 = "stablehlo.abs"(%149) : (tensor<20x20xf32>) -> tensor<20x20xf32>
      %151 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %152 = "stablehlo.broadcast_in_dim"(%151) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %153 = "stablehlo.select"(%146, %150, %152) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %154 = "stablehlo.select"(%146, %147, %arg18) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %155 = "stablehlo.multiply"(%arg29, %135) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %156 = "stablehlo.subtract"(%155, %arg23) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %157 = "stablehlo.multiply"(%arg27, %137) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %158 = "stablehlo.subtract"(%156, %157) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %159 = "stablehlo.broadcast_in_dim"(%129) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %160 = "stablehlo.multiply"(%arg25, %159) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %161 = "stablehlo.add"(%158, %160) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %162 = "stablehlo.multiply"(%arg30, %135) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %163 = "stablehlo.subtract"(%162, %arg24) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %164 = "stablehlo.multiply"(%arg28, %137) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %165 = "stablehlo.subtract"(%163, %164) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %166 = "stablehlo.broadcast_in_dim"(%129) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %167 = "stablehlo.multiply"(%arg26, %166) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %168 = "stablehlo.add"(%165, %167) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %169 = "stablehlo.multiply"(%154, %168) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %170 = "stablehlo.subtract"(%161, %169) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %171 = "stablehlo.divide"(%170, %143) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %172 = "stablehlo.select"(%146, %171, %arg31) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %173 = "stablehlo.subtract"(%172, %arg31) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %174 = "stablehlo.abs"(%173) : (tensor<20x20xf32>) -> tensor<20x20xf32>
      %175 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %176 = "stablehlo.broadcast_in_dim"(%175) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %177 = "stablehlo.select"(%146, %174, %176) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %178 = "stablehlo.abs"(%140) : (tensor<20x20xf32>) -> tensor<20x20xf32>
      %179 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %180 = "func.call"(%179) <{callee = @integer_pow}> : (tensor<f32>) -> tensor<f32>
      %181 = "stablehlo.broadcast_in_dim"(%180) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %182 = "stablehlo.compare"(%178, %181) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %183 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %184 = "stablehlo.broadcast_in_dim"(%183) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %185 = "stablehlo.multiply"(%arg23, %184) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %186 = "stablehlo.select"(%182, %185, %arg23) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %187 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %188 = "stablehlo.broadcast_in_dim"(%187) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %189 = "stablehlo.multiply"(%140, %188) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %190 = "stablehlo.select"(%182, %189, %140) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %191 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %192 = "stablehlo.broadcast_in_dim"(%191) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %193 = "stablehlo.multiply"(%arg24, %192) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %194 = "stablehlo.select"(%182, %193, %arg24) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %195 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %196 = "stablehlo.broadcast_in_dim"(%195) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %197 = "stablehlo.multiply"(%143, %196) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %198 = "stablehlo.select"(%182, %197, %143) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %199 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %200 = "stablehlo.broadcast_in_dim"(%199) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %201 = "stablehlo.multiply"(%arg29, %200) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %202 = "stablehlo.select"(%182, %201, %arg29) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %203 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %204 = "stablehlo.broadcast_in_dim"(%203) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %205 = "stablehlo.multiply"(%arg30, %204) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %206 = "stablehlo.select"(%182, %205, %arg30) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %207 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %208 = "stablehlo.broadcast_in_dim"(%207) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %209 = "stablehlo.multiply"(%161, %208) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %210 = "stablehlo.select"(%182, %209, %161) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %211 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %212 = "stablehlo.broadcast_in_dim"(%211) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %213 = "stablehlo.multiply"(%168, %212) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %214 = "stablehlo.select"(%182, %213, %168) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %215 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %216 = "stablehlo.broadcast_in_dim"(%215) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %217 = "stablehlo.compare"(%153, %216) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %218 = "stablehlo.and"(%arg17, %217) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
      %219 = "stablehlo.select"(%arg17, %154, %arg18) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %220 = "stablehlo.select"(%arg17, %153, %arg19) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %221 = "stablehlo.select"(%arg17, %132, %arg20) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %222 = "stablehlo.select"(%arg17, %135, %arg21) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %223 = "stablehlo.select"(%arg17, %190, %arg23) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %224 = "stablehlo.select"(%arg17, %198, %arg24) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %225 = "stablehlo.select"(%arg17, %186, %arg25) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %226 = "stablehlo.select"(%arg17, %194, %arg26) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %227 = "stablehlo.select"(%arg17, %202, %arg27) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %228 = "stablehlo.select"(%arg17, %206, %arg28) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %229 = "stablehlo.select"(%arg17, %210, %arg29) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %230 = "stablehlo.select"(%arg17, %214, %arg30) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %231 = "stablehlo.select"(%arg17, %172, %arg31) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      "stablehlo.return"(%218, %219, %220, %221, %222, %129, %223, %224, %225, %226, %227, %228, %229, %230, %231) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>) -> ()
    }) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>) -> (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>)
    %73 = "stablehlo.multiply"(%72#1, %39) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %74 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %75 = "stablehlo.broadcast_in_dim"(%74) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %76 = "stablehlo.subtract"(%75, %73) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %77 = "stablehlo.not"(%28) : (tensor<20x20xi1>) -> tensor<20x20xi1>
    %78 = "stablehlo.and"(%43, %77) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %79 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %80 = "stablehlo.broadcast_in_dim"(%79) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %81 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %82 = "stablehlo.broadcast_in_dim"(%81) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %83 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %84 = "stablehlo.broadcast_in_dim"(%83) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %85 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %86 = "stablehlo.broadcast_in_dim"(%85) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %87:7 = "stablehlo.while"(%78, %5#0, %80, %82, %7, %84, %86) ({
    ^bb0(%arg8: tensor<20x20xi1>, %arg9: tensor<20x20xf32>, %arg10: tensor<20x20xf32>, %arg11: tensor<20x20xf32>, %arg12: tensor<20x20xf32>, %arg13: tensor<20x20xf32>, %arg14: tensor<20x20xf32>):
      %125 = "stablehlo.constant"() <{value = dense<false> : tensor<i1>}> : () -> tensor<i1>
      %126 = "stablehlo.reduce"(%arg8, %125) <{dimensions = array<i64: 0, 1>}> ({
      ^bb0(%arg15: tensor<i1>, %arg16: tensor<i1>):
        %127 = "stablehlo.or"(%arg15, %arg16) : (tensor<i1>, tensor<i1>) -> tensor<i1>
        "stablehlo.return"(%127) : (tensor<i1>) -> ()
      }) : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
      "stablehlo.return"(%126) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg1: tensor<20x20xi1>, %arg2: tensor<20x20xf32>, %arg3: tensor<20x20xf32>, %arg4: tensor<20x20xf32>, %arg5: tensor<20x20xf32>, %arg6: tensor<20x20xf32>, %arg7: tensor<20x20xf32>):
      %101 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %102 = "stablehlo.broadcast_in_dim"(%101) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %103 = "stablehlo.add"(%arg2, %102) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %104 = "stablehlo.divide"(%arg5, %103) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %105 = "stablehlo.multiply"(%arg6, %104) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %106 = "stablehlo.multiply"(%arg3, %arg5) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %107 = "stablehlo.multiply"(%103, %103) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %108 = "stablehlo.divide"(%106, %107) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %109 = "stablehlo.subtract"(%105, %108) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %110 = "stablehlo.add"(%arg7, %109) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %111 = "stablehlo.divide"(%arg5, %103) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %112 = "stablehlo.multiply"(%arg3, %111) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %113 = "stablehlo.add"(%arg4, %112) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %114 = "stablehlo.divide"(%112, %113) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %115 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %116 = "stablehlo.broadcast_in_dim"(%115) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %117 = "stablehlo.compare"(%114, %116) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %118 = "stablehlo.and"(%arg1, %117) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
      %119 = "stablehlo.select"(%arg1, %103, %arg2) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %120 = "stablehlo.select"(%arg1, %112, %arg3) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %121 = "stablehlo.select"(%arg1, %113, %arg4) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %122 = "stablehlo.select"(%arg1, %arg5, %arg5) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %123 = "stablehlo.select"(%arg1, %109, %arg6) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %124 = "stablehlo.select"(%arg1, %110, %arg7) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      "stablehlo.return"(%118, %119, %120, %121, %122, %123, %124) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>) -> ()
    }) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>) -> (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>)
    %88 = "stablehlo.multiply"(%87#3, %39) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %89 = "stablehlo.divide"(%88, %5#0) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %90 = "stablehlo.select"(%28, %76, %89) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %91 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %92 = "stablehlo.broadcast_in_dim"(%91) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %93 = "stablehlo.select"(%13, %92, %90) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %94 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %95 = "stablehlo.broadcast_in_dim"(%94) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %96 = "stablehlo.select"(%16, %95, %93) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %97 = "stablehlo.or"(%23, %10) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %98 = "stablehlo.constant"() <{value = dense<0x7FC00000> : tensor<f32>}> : () -> tensor<f32>
    %99 = "stablehlo.broadcast_in_dim"(%98) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %100 = "stablehlo.select"(%97, %99, %96) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    "stablehlo.custom_call"(%100, %6) <{call_target_name = "check.expect_almost_eq", has_side_effect = true}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> ()
    "func.return"(%100) : (tensor<20x20xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<20x20xf32>, tensor<1x20xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %3 = "stablehlo.constant"() <{value = dense<"0x1C75ED3D5200F33F3624F53FCBAAC03D813E78BF1F2616BF4F0F3EBFA01ED4BF82E58B403F93374039DE2CC0453949C02E8F88BE43B9C7C032F1CCBE39AF1840B8DE6A40F82FF0BF0D94E63F9A34B1BFE23273C00757DA3ED2A7DABF52A18240D935DC3F10787640427D53C074295B3F752A0C40BC2CFA3F5F0576C02226893FAC579340A6735AC0416D54BFD96406BD4925C83F5C6520C05CED9FBF9C5F10407F9608C0CF85793E0B64FA3F82A78DC027602540B8F2844007377340EADA59BF353F1BC06E7BB740925A973EEEB147C04C36FA4019345CBF258DE4403D2DF4BE2036B3BD62E8FDBFF65A88C05900C5BFD39D384023A8AA3FC224CABE4C8104C03270F23F2EBAB93DBFEE7BC03EF823BF3EA0EEBFF6A859BF53EAED3FD4E95F4035FC7E3F40F2FFBFA912314071006F3F6D8963C00D4104403CF3014148832940EDD999BFC28C3BC060D63BC0A23615405ED74C40AA41034162FD8B408D1CB93E60A487C07C1889C0F43CFCBE78F327C0941068402FEB1F40AE9150BF975C2E404B5A893D79AD69C08019983F9169CD3F66B35340B7B0E4BFBE4C874093E2C63E26B7DABF06F7ACBF3A710D3C348E523F6CC438409F7932C0C3D9813E7C5F81BF358E90C0503E3EBF20D041C0195CD63F53533640B1F60FBFDFBF4A405AB666400AF5043E729A58BF562356C03BE03ABFBCE2F13EDDB49BBFF52DA640FE9E443F385EFFBF45C91540884F593F58EC4ABF76FB67C0E466E43F57030FC0B4685ABE2AE4274035A58240E83AAEBFBFE0DA4063F5C33F11C2C6BFF4584640608DE13E5C7201C0D33940404CA782C03F4EC7BF874175C06C4DDBBF8E764BBFCBE04C3E73B017BF76B1A8BF992D3FC0153C3F403DAC133ECA11A83EEE2E54BF86AA813F0E9E5FC09A33E1BF874E78403E967B3F4068E34058690FC02443BA3FA1A8D33E28438E400C24F940980D0C3E6C5FC840B0DEBBBFB25550408513C2407E913E40AB473C3F088534C017C5F7BEF6002E409A007DC0D4CFB7C0CDD904C0811528415D9FA93F80CBC5BE34BF5D404DC841BFFE59E73FBC6F3FC0066B953E4E48E3BF4A1AEEBE3F0916406BB149C051B025C07D06D1BF32263F400B1E21C097EDA8C0CCEBB4407B92A9BD1B9E6140496492C0F892DD4070F7CBC0B6216D40E95250408F544D40E214AC3F78EB16C020DEBDBFEFEF283E0A9BAB4018763FBFF0637D3E367957C0C032D4BF6A4907406ECE59408A0C67C0EFBB973FB06B863F561291C0A9C489C0DE39FD4069184BC0632C86C0FFDC09BF91A61D4017DC924037C7CE3FEF42053F6232623ECC8F15C0192C4A403BBD743F252AAEBF25990B403840083F1990394064DE91BDFEA100C07827F03EAE89BA3F01CE0A40F7431EC088725A3F11C919C0F3BA833FD90593BED3CF4EC032E51E4071281540A2B1963E65D8CB3EA858CF40665472BF0C18A140C181F73FF3DA9DC08657B83E8E4A47C009E442C076F4183F0F98643E5F7A1D40B604303F80D21140231DDE40EE20FEC073FFA03F7C4D903FC02173400687DA3F7EDD854045A635C087124ABFE60D9A3FCCA2CEBFBD3AA1BFA3ED74C01D48F73F84A9CB40F11300409FABD43ED476203F38DA263F48625FC06B3504408D4E64C011DEFEBDA10083BFBDB709C031D08ABFBA2C023F2DF4B73DC81890C0DED45B40AAF834C0F9D727409EAAD33FC01E64C03E03FEBFFA51F2BFED90B7BCF499273CF9877CC05B26C4BE7E6EE5BF00D24D40881CFF3FA2FC14C0D465ACC0D77CB8BFBFD91C40E1AE30BE76B588C04E2C20C027D2A0BF2C28EA3F9D0C9EBFB11ACE3EB8EBE13FC00018C0F0050CC052064B40B8D7393E85B7B63F8930CC3F295B81BFFA193D3FDC0294BE18A04340ED954E3F176785BE89B02FC017BC05C091B46FC0068170BFF10E8EC06DCA113FC3F1A23F1587B4BFD4CDCE3F4E278E40F78019403E33DCBFB69FA24013F78FC0847BAD40E702D7BF993DC6C047EED5BE6BDD413FFC7C9F40C051FCBDA292F2BFE80A9D3FD9788E4099E537C0676642C07BEE9540C8C8B740636169C0285DC6C09C817C3F3EC559C0EA6CA7C0709320400A91ABBF2E85CEBF539111C0200080BF96637340EA0A86BF74915940E252573E234BABBE6E606D3E4BC9B83FD1B36FC06BC6B0BEDEBCA2BF55E51ABF578E00C05DE48FBFCBDF9FC01FB782C0D0AD9740585112403334BFBFF2EAD2405F425640F97A36C0A1F8364045F0E53FFD352E40657B223F62B507BE"> : tensor<20x20xf32>}> : () -> tensor<20x20xf32>
    %4 = "stablehlo.constant"() <{value = dense<[[4.39382219, 0.189328432, 1.12942672, 8.07508659, -0.368767828, -0.476076275, -2.56633258, -1.27846646, -2.41863656, 2.14068699, -0.300600946, 5.38453531, -2.61912084, -0.976270675, -0.512060165, -0.295296222, -7.70805025, 6.2874341, -4.14961195, 1.55763531]]> : tensor<1x20xf32>}> : () -> tensor<1x20xf32>
    "func.return"(%3, %4) : (tensor<20x20xf32>, tensor<1x20xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %2 = "stablehlo.constant"() <{value = dense<"0x19E97F3F4D7FA83C59ADAC3EBAFF7F3F0000C07F0000C07F0000C07F0000C07F0000C07F9280C93E0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F9075063F0000C07F75D4743F0000C07F0000C07F0000C07F0000C07F0000C07F64C2243F0000C07F0A9C7E3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FD1C2C43E0000C07F9460353F8946A63E0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F5DB1F63C0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FF8CA543F4C13A83D0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FD8E4593F0000C07F0000C07F0000C07F0000C07F0000C07F6E277C3F0000C07F01E08F3E0000C07F0000C07F0000C07FE4BB7E3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FA42B173F1CCD433F0000C07FF14DA43CACFD7F3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FF4F2E63D9DE57F3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FABD7083F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FC13F5D3F0000C07FCB0BC23AC250773F0000C07F1E78BF3D09FD7F3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FEAF07F3F0000C07F0000C07F0000C07F0000C07F0000C07F1BF47F3F0000C07F8C0D493F0000C07F0000C07FEDF0073D90EC7F3F0000C07F0000C07F0000C07F0000C07F0000C07F76F6133B0000C07F00D1D03E0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FB681873E0000C07F0000C07F0000C07F4508743E0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FC54E733F0000C07F0000C07FBB74AA3E0000C07FC63C5D3D0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F06694E3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FB4570E3E0000C07F306CEB3DB412283F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F611B003F0000C07FE7E77B3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FB8536A3F473E543F0000C07F0000C07FA3FC7F3F0000C07F0000C07F0000C07F0000C07F0000C07F1069603F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F1FEBF63E0000C07F2C73163F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F7212E53B0000C07F4FE97D3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F7C0CAA3E8C63463F0000C07F0000C07F0000C07F0000C07F0000C07F87B31C3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FBBF25D3F26181C3D0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FE293783F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F4B69723F0000C07F152E663FA5887F3F0000C07F0000C07F0000C07F0000C07F0000C07FD7113E3F0000C07F3D6A7F3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F9266D03E4F2E113F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F3EA92F3D0000C07F0000C07FAEC4233C5976573F0000C07F0000C07F0000C07F0000C07F0000C07FDBCBF93E0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F1AFA7F3F0000C07F0ACD783FB17C783F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F65DCAF3D0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FCCA2763F0000C07F0000C07F"> : tensor<20x20xf32>}> : () -> tensor<20x20xf32>
    "func.return"(%2) : (tensor<20x20xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<f32>) -> tensor<f32>, sym_name = "integer_pow", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<f32>):
    %0 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %1 = "stablehlo.divide"(%0, %arg0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "func.return"(%1) : (tensor<f32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

