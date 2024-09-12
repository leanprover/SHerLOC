"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %5:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x20xf32>, tensor<20x20xf32>)
    %6 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xf32>
    %7 = "stablehlo.broadcast_in_dim"(%5#0) <{broadcast_dimensions = array<i64: 0, 1>}> : (tensor<1x20xf32>) -> tensor<20x20xf32>
    %8 = "stablehlo.compare"(%7, %7) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %9 = "stablehlo.compare"(%5#1, %5#1) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %10 = "stablehlo.or"(%8, %9) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %11 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %12 = "stablehlo.broadcast_in_dim"(%11) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %13 = "stablehlo.compare"(%5#1, %12) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %14 = "stablehlo.constant"() <{value = dense<0x7F800000> : tensor<f32>}> : () -> tensor<f32>
    %15 = "stablehlo.broadcast_in_dim"(%14) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %16 = "stablehlo.compare"(%5#1, %15) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %17 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %18 = "stablehlo.broadcast_in_dim"(%17) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %19 = "stablehlo.compare"(%5#1, %18) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %20 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %21 = "stablehlo.broadcast_in_dim"(%20) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %22 = "stablehlo.compare"(%7, %21) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %23 = "stablehlo.or"(%19, %22) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %24 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %25 = "stablehlo.broadcast_in_dim"(%24) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %26 = "stablehlo.compare"(%5#1, %25) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %27 = "stablehlo.compare"(%5#1, %7) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %28 = "stablehlo.and"(%26, %27) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %29 = "stablehlo.log"(%5#1) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %30 = "stablehlo.multiply"(%7, %29) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %31 = "stablehlo.subtract"(%30, %5#1) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %32 = "chlo.lgamma"(%7) : (tensor<20x20xf32>) -> tensor<20x20xf32>
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
    %47 = "stablehlo.subtract"(%46, %7) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %48 = "stablehlo.add"(%5#1, %47) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %49 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %50 = "stablehlo.broadcast_in_dim"(%49) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %51 = "stablehlo.add"(%48, %50) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %52 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %53 = "stablehlo.broadcast_in_dim"(%52) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %54 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %55 = "stablehlo.broadcast_in_dim"(%54) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %56 = "stablehlo.add"(%5#1, %55) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %57 = "stablehlo.multiply"(%51, %5#1) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %58 = "stablehlo.divide"(%56, %57) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %59 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %60 = "stablehlo.broadcast_in_dim"(%59) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %61 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %62 = "stablehlo.broadcast_in_dim"(%61) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %63 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %64 = "stablehlo.broadcast_in_dim"(%63) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %65 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %66 = "stablehlo.broadcast_in_dim"(%65) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %67 = "stablehlo.negate"(%5#1) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %68 = "stablehlo.multiply"(%58, %67) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %69 = "stablehlo.subtract"(%66, %68) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %70 = "stablehlo.divide"(%69, %57) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %71 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %72:15 = "stablehlo.while"(%44, %58, %60, %47, %51, %71, %56, %57, %53, %5#1, %62, %64, %66, %67, %70) ({
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
    %87:7 = "stablehlo.while"(%78, %7, %80, %82, %5#1, %84, %86) ({
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
    %89 = "stablehlo.divide"(%88, %7) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
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
  "func.func"() <{function_type = () -> (tensor<1x20xf32>, tensor<20x20xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[-1.6881727, 2.78302264, 1.72012115, 6.96233272, -0.654121696, 5.66036797, 0.782327175, 1.67816162, -2.48140669, -4.60913324, 7.00846099, 2.21609473, -3.05883622, -0.913837909, -1.77270198, 2.91084933, 1.2070893, 1.45224094, -1.30566072, -6.07208776]]> : tensor<1x20xf32>}> : () -> tensor<1x20xf32>
    %4 = "stablehlo.constant"() <{value = dense<"0x653003C045D2BABFF0E40F40D01FF03FE946B43E7120153E94E9593F680600C02DA4DD3EBED3223F76E731C0978F26407051134045088DC07D5848BF964C813F79B3F03F8D793F40E1ADD93CDE463BBE0A0C7A40EE8362C0B67E0D4014F2B7BE7D0DE7BFBF5BE5BFB5072A3F943405C0024C29C0648CE7C0F8E735408023144069EB4340013D12C09F380140C124FC3F467C53C0BE7DBA3F47E1E440A95001405BA6B4C08BC7A53EB19098C044C797C00D003E3F84E062C00C4D4840E6994C40A72308C051E32D4079BBDDBF4CA21F402D438C3F3B4C5F40848BC7BF80BC74C015A300C0305E1BC1A2C961BE83A2C33F8F96E4C0955F91408292D6BF1E2C16BE338FEFBF39AB8640CFD3E7BFC7905BC077C973C0271CCCBFF5B98CC0F80C72BE47476DC089D9F540724775BF845A4740873074BF4A41CCBF07C8573E4E7838C00B7BB3C0C2D3853E3AD0923FBB187FC0BEC1123F1CB52F3F3E3CA63FA4B706C0FE9BD43F0DCF493E1DAD2FBEAAD343BF40113F40EA15D63F6B880440E39DA5BDEC9CB6BED4589D4004DB883E58B19FBF16902640768E50C0E0342F40F9801B3F55271B4038340EBFEE2303402A877B3F2722B640A06BF0BF12C435407283C73EFEB74140B5FD93BF16FA10C0D21D15C08BDC92BE5EEF294056D23DC0C34924BFA5F633402F620E409D924ABFC1922E3F220FDFBF4A77F5BD992A12C0188CBCBFD97498C0879EE5BE050334404E3C22C00BE10740B26F983F6D5F14C08CE614C02562C73FCDF271BF72588AC075BCD6BF1BDEF5BE55C523404D44FA3D66B2014006DECA40E91392BE606D30BF6D2582C085BC88C0078616BFCB0DA9408C157F3FAFB80A40D29D7E40F5738CBFD8AE8E403C7F0CC098CCE7BEB887653FA07B7F40835A36403DA0713FABF3F7BE868D16C055238340D86685C05F370B4062DFBABF72FB4DC0668385BF5407D0BF30D01341B665404088E7284065444F409476E6BF712BDFC01ED45A3FC0E088C0C30E864027905E40D20DAABF69D29FBEB320E6BE3F773E4031F06D3F524C3A3E1FE2DDBFC698D63E5CA5973ECA12713EEEF72740F0CF38C0DDFA1E40EA198F3F379FF83F8F4A6CC06D5C48403DFBC53F52B80AC0914F1D3E61FAC63D6571A7BF8D2E824070D3523FC9B5BC406353D33D71BFF5BED5F6FFC0DA327AC03747EBC0891F474080345A3FFAD751C0AE3902C0851D42BE395DBFBF8C18203EF6A14B3E984335BE972919413D9682C052E82540A2CC09C062E1FD3F41D051C0611D5740BD1666C0105A03C1CF78B9BFC7665840E067AF3F7E43F43F355E13402593A43F2738B1C0AA9AEBBFB227C3BF0B7FB4BFB21080C0E136683F41CEB4409013C43F597C60C03E2B87BF3F9875BFC4F830BF155DCB405B2CA940328219C0F22492BEE43443C0CE87AEBFA2B6A140434BA5BF0F2F0CC00D271CC07FF71CC0B285373FA0061BC0BAA253C068878BC00412BEBE194B933FD590BD3F1D7BD54032E610C02283263F285D6CBF08B8BC3F27A45F4028946AC0285D09C1E258693FEE2238C00D188F4091AA284082D50C40101632BF31AECB3F4C04AE4066D6F1BF314F5F401EDCF93F81194A3FADEF6040809894C068310A4013E34FBE9036893FDBC2C23F4D78CC3F98F511406E3BFB3DE74DAA3E19F89A4079EF98BF281BDE3E99E30DC08A78C23FBCB4D1BFBDC17240DBC9A73F910019C0F97DD3BD5AD1D43FEB91994095485ABE01DB8B402F157DBF3451DFBE44296C408D4515BF3FDFA2C07D4887C0E12A783F39B73BC0B31FD6C01A3090BEE65DACBFEA4B31C0C93AE03F8BFA27BF36E498C09B5F173F6901CD3E1AD723408BF7F94066835740135EFA40C8968AC01FDCA5BF361D1941A98FBBBF8A6B803F811D9FC0A4D1B13F42DC09403B3F8E407AF750C03F3AFE3F6D6E6840ADD400408CBBC7C0237766C095D533C0A70173C09A6F8C4028E881C0F7B9F9BFDF5C063F06A295BF7E457B40B0A095BF229501C0F6FBA4C07F2C18C03C88E7C090D1733E40590B3F447969404A696CBFE4B45AC07A537C3FF16858BF95C180401F7FCA4024E289C08433733F240BA7C01D973F405C925DBF9BBB30C0360E7BBF8D13B9BFF911BD3F05C59E402F0866BF490F2DBF51AC14C0F4E8D4409555393F822C4340B7E01DC0D88D3A3FE67853BDC8725BBF3D795E40A148ECBF40F63C3DA5F62ABFBD9B8EC08017E03EC6788A3F83AA00C048B087C0D7721FC03957C3BF9FBD9BC06B180740"> : tensor<20x20xf32>}> : () -> tensor<20x20xf32>
    "func.return"(%3, %4) : (tensor<1x20xf32>, tensor<20x20xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %2 = "stablehlo.constant"() <{value = dense<"0x0000C07F0000C07F30623B3FB06C5E3B0000C07F8E21353303EB2C3F0000C07F0000C07F0000C07F0000C07FDB2C2E3F0000C07F0000C07F0000C07FCC11BD3DDA5F4B3F1E0B653F0000C07F0000C07F0000C07F0000C07F475E393F0000C07F0000C07F0000C07FA6BB183F0000C07F0000C07F0000C07FC28DD43C563C1D3F0000C07F0000C07F0000C07FE210AC3E0000C07F35911C3F0000C07F0000C07F0000C07F98EAF33B0000C07F0000C07F0000C07F0000C07F3104793F318F613F0000C07F0000C07F0000C07F1618283F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F254C5C3F0000C07F0000C07F0000C07FABED983E0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FB9681F3F0000C07F0000C07F0000C07F0000C07F0000C07F4DA08C3B44A1D03E0000C07F0000C07F8D7C3539C9DA4D3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F5B497B3F0000C07F0000C07F0000C07F0000C07FEB5E513F2D9983360000C07F0000C07F31E9693F5595B63E0000C07F0000C07F36CBD33C01151E3D0000C07F0000C07F0000C07F0000C07F0000C07F6EC65B3F0000C07F0000C07F0000C07FBF5FE03E0000C07F11070A370000C07F0000C07F0000C07F0000C07F0000C07F0000C07FCD71CA3C0000C07F0000C07F0000C07F0000C07F0000C07F963C393F0000C07F0000C07F0000C07F0000C07F2493063F8DF8813C8C92A73B0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FE2E58E3EEA1A553E0000C07F0000C07F0000C07FD6AA553F0000C07F0000C07F0000C07F0000C07F0000C07FB0F3C03D0000C07F0000C07F0000C07F0000C07F6FB76C3F0000C07F0000C07F0000C07F0000C07FDA9E7F3F0000C07F0000C07F0000C07F0000C07F0000C07FAF1AC43E0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F85044E3AD6A8863E0000C07F0000C07F0000C07F6907D8317A5E2F3F0000C07F0000C07F0000C07FB82CA83E0000C07F9B29683F0000C07F0000C07F0000C07F406EA1390000C07FE0EDF73D0000C07F921D183F41B8323E0000C07F0000C07F0000C07F0000C07F6459463F0000C07F0000C07F0000C07F0000C07F0000C07F1443443D0000C07F0000C07F0000C07F0000C07F4D9F4B3F0000C07F0000C07F0000C07F33877A3F0000C07F0000C07F0000C07FF8D0653DC4F4AA3E0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F63C86F3F3E2F0B3F0000C07F0000C07F0000C07F0000C07FB9197E3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F19F85B390000C07F4AA4343F0000C07F42095D3E0000C07F0000C07FCCE2833D0000C07F0000C07F0000C07F0000C07F6206563FCEF7653F8E5E4A3F0000C07F0000C07F0000C07F0000C07F74DC663F0083893B0000C07F39263B3E0000C07F6568393F0000C07F0000C07F2AE2813ACA3CD13E0000C07F0000C07F0000C07F26485F3F0000C07FD652383E0000C07F0000C07F0000C07F5042453F40D3EF3E0000C07F0000C07FC7033D3C26C87E3F0000C07F0000C07F0000C07F0000C07F55205A3F0000C07F0000C07F0000C07F7A96AC3D0000C07F0000C07F0000C07F0000C07F0000C07FC4559C3E0000C07F0000C07F0000C07F17C82E373C39733FD17F7F3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FA48A2D3F7834483F0000C07F0000C07F0000C07F77EE3E3F25DB2D3F0000C07F0000C07F0000C07F0000C07F7306753F0000C07F0000C07F96C6B4350000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F1785AC380000C07F0858883ED6BF7F3F0000C07F0000C07F0000C07F47CF063D0000C07F0000C07F0000C07F0000C07F90EA4E3EA32C7D3F0000C07F0000C07F0000C07F0000C07FE321573DAACE5B3F0000C07F0000C07F0000C07F0000C07FE6EE673F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F"> : tensor<20x20xf32>}> : () -> tensor<20x20xf32>
    "func.return"(%2) : (tensor<20x20xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<f32>) -> tensor<f32>, sym_name = "integer_pow", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<f32>):
    %0 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %1 = "stablehlo.divide"(%0, %arg0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "func.return"(%1) : (tensor<f32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

